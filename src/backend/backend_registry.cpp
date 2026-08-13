// backend_registry.cpp — the ONE definition of backend::make_backend /
// backend::available_backends, shared by every vendor. See backend_registry.h.
//
// This TU replaces the three near-identical per-vendor copies
// (cpu/backend/cpu_backend_registry.cpp, nvidia/backend/nv_backend_registry.cpp,
// apple/backend/apple_backend_registry.cpp), each of which hard-coded its own name
// matching, alias list and "auto-detect => me" rule. Those files are now pure
// registration (one BackendRegistrar each), so linking several of them into ONE
// binary is not just legal but the point: --backend picks among them at runtime.

#include "turbo_ocr/backend/backend_registry.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "turbo_ocr/base/log/logger.h"

namespace turbo_ocr::backend {
namespace {

struct Entry {
  std::string name;
  std::vector<std::string> aliases;
  int priority = 0;
  BackendFactory factory = nullptr;
  AutoUsableProbe auto_usable = nullptr; // auto-detect-only; see the header

  [[nodiscard]] bool matches(std::string_view n) const {
    if (name == n) return true;
    return std::any_of(aliases.begin(), aliases.end(),
                       [&](const std::string &a) { return a == n; });
  }
};

// Entries are heap-stable (unique_ptr) because available_backends() hands out
// string_views into Entry::name — a plain vector<Entry> would dangle them on
// reallocation.
using Registry = std::vector<std::unique_ptr<Entry>>;

// Construct-on-first-use: a registration TU's static initializer may run before
// any namespace-scope object in THIS TU would have been constructed, so the
// storage must not itself be a namespace-scope object (static init order fiasco).
Registry &registry() {
  static Registry r;
  return r;
}
std::mutex &registry_mutex() {
  static std::mutex m;
  return m;
}

// One row of the snapshot make_backend() works from once the registry lock is
// released. `name` is COPIED rather than viewed into Entry::name so the row is
// self-contained regardless of what happens to the registry meanwhile.
struct Candidate {
  std::string name;
  BackendFactory factory = nullptr;
  AutoUsableProbe auto_usable = nullptr;
};

// Collect the entries make_backend() would consider, under the lock and doing
// nothing else under it. `name` empty => every entry (auto-detect walks them in
// priority order); otherwise the entries whose canonical name or alias matches.
std::vector<Candidate> snapshot_candidates(std::string_view name) {
  const std::lock_guard<std::mutex> lk(registry_mutex());
  const Registry &r = registry();
  std::vector<Candidate> out;
  out.reserve(r.size());
  for (const auto &e : r)
    if (name.empty() || e->matches(name))
      out.push_back({e->name, e->factory, e->auto_usable});
  return out;
}

} // namespace

void register_backend(std::string_view name,
                      std::initializer_list<std::string_view> aliases,
                      int priority, BackendFactory factory,
                      AutoUsableProbe auto_usable) {
  if (name.empty() || factory == nullptr) return;
  const std::lock_guard<std::mutex> lk(registry_mutex());
  Registry &r = registry();
  for (const auto &e : r)
    if (e->name == name) return; // idempotent

  auto e = std::make_unique<Entry>();
  e->name.assign(name);
  e->auto_usable = auto_usable;
  e->aliases.reserve(aliases.size());
  for (std::string_view a : aliases) e->aliases.emplace_back(a);
  e->priority = priority;
  e->factory = factory;

  // Keep the registry sorted by DESCENDING priority so both auto-detect and the
  // available_backends() listing have a deterministic, device-preference order
  // regardless of static-init order (which is link-order dependent).
  const auto pos = std::find_if(
      r.begin(), r.end(),
      [priority](const std::unique_ptr<Entry> &x) { return x->priority < priority; });
  r.insert(pos, std::move(e));
}

std::unique_ptr<Backend> make_backend(std::string_view name) {
  // Snapshot first, then run vendor factories with the lock RELEASED. Two
  // reasons, both real:
  //   * registry_mutex() is a plain (non-recursive) std::mutex, and a factory
  //     calling back into this TU is entirely plausible — registering a
  //     sub-backend it just discovered, or dumping available_backends() as a
  //     diagnostic on failure. Under the old code that self-deadlocked.
  //   * a factory does full device initialisation (CUDA context creation, Metal
  //     device probe, Level Zero / OpenVINO plugin load). Holding a
  //     process-global lock across that — for every candidate in turn — is
  //     unbounded time under a lock two Python threads can reach concurrently
  //     (src/service/python/bindings.cpp constructs a Backend per Pipeline).
  const std::vector<Candidate> candidates = snapshot_candidates(name);

  if (name.empty()) {
    // Auto-detect: highest-priority vendor that is actually usable here. A
    // vendor whose device is absent may legitimately throw or return nullptr
    // from its factory (e.g. AppleBackend on a Metal-less machine), so keep
    // walking down to the host backend rather than failing the whole server.
    //
    // Every fall-through is LOGGED. Silently swallowing them is how a CUDA
    // driver/library mismatch, a Metal device that would not open, or a failed
    // Level Zero init turns into a host-backend server — an order-of-magnitude
    // throughput loss on a machine that has the hardware, with zero operator
    // signal, because the only startup line afterwards names whoever WON.
    for (const auto &c : candidates) {
      // Registrar-supplied usability probe: lets a vendor whose factory never
      // declines (it degrades internally) still yield auto-detect to a
      // lower-priority vendor with real hardware. Named selection below does
      // NOT consult it — the degraded configuration stays reachable.
      if (c.auto_usable && !c.auto_usable()) {
        TOCR_LOG_INFO("Backend declined auto-detect (no device), trying next",
                      "backend", std::string_view(c.name));
        continue;
      }
      try {
        if (auto b = c.factory()) return b;
        TOCR_LOG_INFO("Backend not usable here (factory returned null), trying next",
                      "backend", std::string_view(c.name));
      } catch (const std::exception &ex) {
        TOCR_LOG_WARN("Backend factory failed during auto-detect, trying next",
                      "backend", std::string_view(c.name), "error", ex.what());
      } catch (...) {
        TOCR_LOG_WARN("Backend factory failed during auto-detect with a "
                      "non-std exception, trying next",
                      "backend", std::string_view(c.name));
      }
    }
    TOCR_LOG_WARN("Auto-detect produced no usable backend", "candidates",
                  static_cast<int>(candidates.size()));
    return nullptr;
  }

  // Explicitly named backend: first entry whose canonical name or alias matches.
  // Deliberately NOT wrapped in try/catch — `--backend nvidia` on a box with a
  // broken driver must fail loudly, not quietly degrade to the host backend.
  if (!candidates.empty()) {
    const Candidate &c = candidates.front();
    if (auto b = c.factory()) return b;
    // Per the registrar contract (backend_registry.h) a factory that returns
    // nullptr and one that throws mean the SAME thing: "compiled in, but not
    // usable on this machine". Only the throwing half used to reach the
    // operator. The other half fell out as nullptr — indistinguishable from
    // this function's "no such backend" answer — so `--backend intel` on a
    // GPU-less box was reported as "not compiled into this build (available:
    // cpu, intel)", an error naming the backend inside its own available list.
    // Both halves now fail loudly, leaving nullptr to mean ONLY "no entry with
    // that name", which is exactly what the caller's message claims.
    TOCR_LOG_ERROR("Named backend is compiled in but its factory declined "
                   "(no device found, or device init failed)",
                   "backend", std::string_view(c.name));
    throw std::runtime_error(
        "backend '" + std::string(name) +
        "' is compiled into this build but is not usable on this machine: its "
        "factory found no device, or device initialisation failed. Pass a "
        "different --backend, or omit it to auto-detect.");
  }
  return nullptr; // no entry with that name — not compiled into this build
}

std::vector<std::string_view> available_backends() {
  const std::lock_guard<std::mutex> lk(registry_mutex());
  const Registry &r = registry();
  std::vector<std::string_view> out;
  out.reserve(r.size());
  for (const auto &e : r) out.emplace_back(e->name);
  return out;
}

} // namespace turbo_ocr::backend
