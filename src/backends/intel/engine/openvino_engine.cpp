// OpenVINOEngine — OpenVINO Runtime IEngine implementation.
//
// TOOLCHAIN: OpenVINO 2024+ (`find_package(OpenVINO)` -> openvino::runtime).
// The device-resident GPU path additionally needs the Level Zero loader so
// ov::RemoteContext can share L0Allocator's context. NOT compilable on the dev
// Mac (no OpenVINO); guarded by TURBO_OCR_HAS_OPENVINO. The guarded-IN branch is
// the authoritative implementation; the guarded-out branch is a
// signature-checked stub whose load() fails, so a build without OpenVINO cleanly
// reports "stage unavailable" instead of pretending to infer.

#include "intel/engine/openvino_engine.h"
#include "intel/memory/l0_allocator.h"

#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(TURBO_OCR_HAS_OPENVINO)
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>
// ZERO-COPY USM BINDING, gated separately from OpenVINO itself.
//
// ov::intel_gpu::ocl::ClContext::create_tensor(type, shape, void* usm_ptr) is
// THE documented way to wrap a caller-owned USM pointer as a remote tensor
// (SharedMemType::USM_USER_BUFFER) — the signature was read out of the installed
// OpenVINO headers, not guessed. But <openvino/runtime/intel_gpu/ocl/ocl.hpp>
// transitively needs the OpenCL C++ headers (CL/cl2.hpp), which ship with the
// Intel OpenCL/oneAPI stack and are absent on a generic host. So it gets its own
// macro: with TURBO_OCR_HAS_OV_USM the engine binds USM pointers in place; the
// rest of the OpenVINO integration compiles and runs either way, and without it
// io_space simply reports Host and I/O stages through pre-sized mirrors.
#if defined(TURBO_OCR_HAS_OV_USM)
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#endif
#endif

namespace turbo_ocr::intel {

// ---------------------------------------------------------------------------
// Device selection (toolchain-independent, so it is outside the guards).
// ---------------------------------------------------------------------------

namespace {
// OV_DEVICE, upper-cased, once. Empty when unset.
const std::string &ov_device_env() {
  static const std::string v = [] {
    std::string s = env::env_or("OV_DEVICE", "");
    for (auto &c : s)
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
  }();
  return v;
}
// The family part of an OV_DEVICE value: "GPU.1" -> "GPU", "GPU" -> "GPU".
std::string ov_device_family(const std::string &s) {
  return s.substr(0, s.find('.'));
}
} // namespace

// The exact string handed to OpenVINO for this device.
//
// OV_DEVICE may name an explicit device INDEX ("GPU.1"), and on a box with both
// an integrated and a discrete card that is the difference between measuring
// two entirely different chips: OpenVINO enumerates BOTH under the GPU plugin
// (verified on a Core Ultra 9 285K + RTX 5090 -> GPU.0 = Intel Xe iGPU,
// GPU.1 = NVIDIA RTX 5090), and a bare "GPU" always resolves to GPU.0. So an
// indexed value is passed through verbatim — but only when it names THIS
// family, so a per-stage override to another device still gets that device's
// plain name rather than an index that belongs to something else.
const char *OpenVINOEngine::device_name(DeviceType d) noexcept {
  const char *base = "CPU";
  switch (d) {
  case DeviceType::CPU: base = "CPU"; break;
  case DeviceType::GPU: base = "GPU"; break;
  case DeviceType::NPU: base = "NPU"; break;
  }
  const std::string &env = ov_device_env();
  if (env.find('.') != std::string::npos && ov_device_family(env) == base)
    return env.c_str();  // static storage: safe to hand back as const char*
  return base;
}

OpenVINOEngine::DeviceType
OpenVINOEngine::device_from_env(DeviceType fallback) {
  const std::string &s = ov_device_env();
  if (s.empty())
    return fallback;
  // Accept an explicit index ("GPU.1"); the family decides the DeviceType and
  // device_name() carries the index through to compile_model().
  const std::string fam = ov_device_family(s);
  if (fam == "CPU")
    return DeviceType::CPU;
  if (fam == "GPU")
    return DeviceType::GPU;
  if (fam == "NPU")
    return DeviceType::NPU;
  // Never silently substitute: an operator who pinned a device and got a
  // different one has no way to tell from the outside. The per-stage override
  // path (intel_backend.cpp::device_for) warns for exactly this reason; the
  // backend-wide knob used to fall through mute.
  TOCR_LOG_ERROR("OV_DEVICE is not a recognized device, using the default",
                 "value", s, "expected",
                 "CPU|GPU|NPU, optionally with an index (e.g. GPU.1)");
  return fallback;
}

// Does the OpenVINO runtime itself enumerate this device?
//
// This is the CORRECT availability question, and it is deliberately NOT
// "L0Allocator::has_device()". That one answers "was this built with SYCL AND is
// there a Level-Zero USM context", which is the question for ZERO-COPY interop,
// not for whether the device can run inference at all. Measured on Core Ultra 7
// 265T / WSL2: the OpenVINO GPU plugin drives the iGPU perfectly well with no L0
// context of ours — it just stages through host memory, which caps() already
// reports honestly as io_space = Host. Gating availability on has_device() made
// an entirely working GPU unreachable.
bool OpenVINOEngine::device_available(DeviceType d) {
#if defined(TURBO_OCR_HAS_OPENVINO)
  try {
    ov::Core core;
    const std::string want = device_name(d);
    for (const auto &have : core.get_available_devices()) {
      // available_devices() reports "GPU", but also "GPU.0"/"GPU.1" on
      // multi-adapter hosts; a prefix match accepts those.
      if (have == want || have.rfind(want + ".", 0) == 0)
        return true;
    }
  } catch (const std::exception &) {
    // A broken/absent plugin must read as "not available", never as a throw
    // escaping the registry during static init.
  }
  return false;
#else
  (void)d;
  return false;
#endif
}

#if defined(TURBO_OCR_HAS_OPENVINO)
namespace {

// Stable key for a primary-input shape. Shapes are small, positive and few, so a
// cheap FNV-1a over the dims is collision-safe in practice; the full shape is
// stored alongside and compared on lookup, so a collision cannot mis-select.
[[nodiscard]] std::uint64_t shape_key(const std::vector<std::int64_t> &s) noexcept {
  std::uint64_t h = 1469598103934665603ull;
  for (std::int64_t d : s) {
    auto v = static_cast<std::uint64_t>(d);
    for (int b = 0; b < 8; ++b) {
      h ^= (v >> (b * 8)) & 0xffu;
      h *= 1099511628211ull;
    }
  }
  return h;
}

[[nodiscard]] std::size_t dtype_size(backend::DType t) noexcept {
  switch (t) {
  case backend::DType::F32: return 4;
  case backend::DType::F16: return 2;
  case backend::DType::I64: return 8;
  case backend::DType::I32: return 4;
  case backend::DType::U8:  return 1;
  }
  return 4;
}

[[nodiscard]] std::size_t elem_count(const std::vector<std::int64_t> &s) noexcept {
  std::size_t n = 1;
  for (std::int64_t d : s)
    n *= static_cast<std::size_t>(d > 0 ? d : 0);
  return n;
}

[[nodiscard]] ov::element::Type to_ov(backend::DType t) {
  switch (t) {
  case backend::DType::F32: return ov::element::f32;
  case backend::DType::F16: return ov::element::f16;
  case backend::DType::I64: return ov::element::i64;
  case backend::DType::I32: return ov::element::i32;
  case backend::DType::U8:  return ov::element::u8;
  }
  return ov::element::f32;
}

} // namespace

struct OpenVINOEngine::Impl {
  // One compiled artefact for one primary-input shape. Everything a run() needs
  // is here, pre-built, so run() only binds and infers.
  struct Variant {
    std::vector<std::int64_t> key_shape; // primary input shape ({} == dynamic)
    ov::CompiledModel compiled;
    ov::InferRequest request;
    // Pre-sized host mirrors, used ONLY in the degraded case where the caller
    // binds a memory space this compiled model cannot accept directly. Sized at
    // prebuild from the compiled IO shapes so run() never allocates.
    std::unordered_map<std::string, std::vector<char>> staging;
    std::unordered_map<std::string, std::vector<std::int64_t>> out_shapes;
    // EXTRA in-flight requests, for the batch-split path in run(). `request`
    // above stays the single-request path; these are only used when a batch can
    // be cut into independent chunks. Sized from the plugin's own
    // OPTIMAL_NUMBER_OF_INFER_REQUESTS (i.e. its stream count), so we ask for
    // exactly the parallelism the compiled model was built with.
    std::vector<ov::InferRequest> pool;
  };

  DeviceType device;
  std::shared_ptr<L0Allocator> alloc;

  ov::Core core;
  std::shared_ptr<ov::Model> model; // parsed once; reshaped per variant
#if defined(TURBO_OCR_HAS_OV_USM)
  // The GPU plugin's context, typed so we can call the USM-pointer overload of
  // create_tensor (ClContext, not the generic RemoteContext).
  ov::intel_gpu::ocl::ClContext remote{nullptr, nullptr};
#endif
  bool has_remote = false;

  std::unordered_map<std::uint64_t, Variant> variants;
  std::unique_ptr<Variant> dynamic_variant;

  std::vector<std::string> in_names, out_names;
  bool loaded = false;
  std::size_t misses = 0;

  Impl(DeviceType d, std::shared_ptr<L0Allocator> a)
      : device(d), alloc(std::move(a)) {}

  [[nodiscard]] backend::DeviceKind io_space() const noexcept {
    return has_remote ? backend::DeviceKind::L0 : backend::DeviceKind::Host;
  }


  // Compile-time configuration for compile_model().
  //
  // DEFAULT hint: LATENCY. This was `throughput` until 2026-08-03, on the
  // strength of rec_tiny's model-only rate under that hint (2144 crops/s on a
  // Core Ultra 7 265T). That number was real but unreachable: the throughput
  // hint partitions the cores into several streams, and it pays off only with
  // several requests IN FLIGHT — while this engine submits ONE synchronous
  // request at a time, which then runs on ONE stream, i.e. a fraction of the
  // machine. Measured end-to-end on FUNSD-50 (i5-13600K, per-stage profile):
  //
  //   hint=throughput             2.4 img/s   rec_infer 317 ms/pg  det_infer 41
  //   hint=throughput + dynbatch  3.3 img/s   rec_infer 205 ms/pg  det_infer 41
  //   hint=latency                5.5 img/s   rec_infer 115 ms/pg  det_infer 17
  //   hint=latency  + dynbatch    5.5 img/s   (identical — split adds nothing)
  //   ORT-CPU reference           4.9 img/s   rec_infer 124 ms/pg  det_infer 30
  //
  // So under the SYNC engine, latency mode (one stream, all cores per request)
  // is strictly better on every stage, and it is what finally puts the native
  // OpenVINO path AHEAD of ORT-CPU on the same silicon. If the engine ever goes
  // truly async (caps().async == true, several requests genuinely in flight),
  // re-measure throughput mode — that is the regime it was designed for.
  //
  //   OV_PERF_HINT   = throughput | latency | none   (default: latency)
  //   OV_NUM_STREAMS = <int>                         (default: unset -> hint decides)
  //   OV_INFER_PRECISION = f32 | f16   (default: f16 on GPU, plugin default on
  //                                    CPU — see the measurements below)
  //   OV_CACHE_DIR   = <path>   persists compiled kernels across runs; GPU
  //                             compiles are seconds each, so this turns a slow
  //                             cold start into a warm one.
  [[nodiscard]] ov::AnyMap compile_config() const {
    ov::AnyMap cfg;

    std::string hint = env::env_or("OV_PERF_HINT", "latency");
    for (auto &c : hint) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (hint == "throughput")
      cfg[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
    else if (hint == "latency")
      cfg[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    // "none" -> leave the plugin default in place.

    if (const int n = env::env_int("OV_NUM_STREAMS", 0, 0, 1024); n > 0)
      cfg[ov::num_streams.name()] = ov::streams::Num(n);

    // Inference precision. DEFAULT f16 ON GPU — measured, not assumed:
    //
    //   rec_tiny  GPU  f16 1629 crops/s  vs f32 1147   (+42%)
    //   det_tiny  GPU  f16 43.7 FPS      vs f32 38.7   (+13%)
    //   end-to-end GPU f16 2.8 img/s     vs 2.5        (+12%)
    //                  peak RSS 1453 MB  vs 2639 MB    (-45%)
    //   F1 85.60% either way — IDENTICAL to two decimal places.
    //
    // Free throughput and nearly half the memory at no accuracy cost, and the
    // memory half matters more on a discrete card with fixed VRAM. GPUs are
    // built for f16; this is the right default for the device.
    //
    // CPU is left alone: the plugin picks f32/bf16 from what the part actually
    // supports, and forcing f16 there is emulated and slower.
    std::string prec = env::env_or("OV_INFER_PRECISION", "");
    if (prec.empty() && device == DeviceType::GPU)
      prec = "f16";
    if (prec == "f16") cfg[ov::hint::inference_precision.name()] = ov::element::f16;
    else if (prec == "f32") cfg[ov::hint::inference_precision.name()] = ov::element::f32;

    return cfg;
  }

  // Compile one variant. `primary` empty => keep the model's own (possibly
  // dynamic) shapes. WARMUP ONLY.
  [[nodiscard]] bool build_variant(const std::vector<std::int64_t> &primary,
                                   Variant &out) {
    auto m = model->clone();
    if (!primary.empty() && !m->inputs().empty()) {
      ov::PartialShape ps(std::vector<ov::Dimension>(primary.begin(), primary.end()));
      std::map<std::string, ov::PartialShape> reshape_map;
      reshape_map[m->input(0).get_any_name()] = ps;
      m->reshape(reshape_map);
    }
    out.key_shape = primary;
#if defined(TURBO_OCR_HAS_OV_USM)
    if (has_remote)
      out.compiled = core.compile_model(m, remote, compile_config());
    else
#endif
      out.compiled = core.compile_model(m, device_name(device), compile_config());
    out.request = out.compiled.create_infer_request();

    // Request pool for the batch-split path. OpenVINO's throughput comes from
    // having several requests in flight across its streams; with ONE synchronous
    // request the plugin runs at streams=1 rates no matter what hint it was
    // compiled with (measured: rec_tiny 476 crops/s at streams=1 vs 2144 with
    // the throughput hint). OPTIMAL_NUMBER_OF_INFER_REQUESTS is the plugin's own
    // answer for how many it wants; OV_NUM_REQUESTS overrides.
    {
      std::size_t want = 1;
      try {
        want = out.compiled.get_property(ov::optimal_number_of_infer_requests);
      } catch (const std::exception &) {
        want = 1;
      }
      if (const int n = env::env_int("OV_NUM_REQUESTS", 0, 0, 1024); n > 0)
        want = static_cast<std::size_t>(n);
      want = std::min<std::size_t>(std::max<std::size_t>(want, 1), 32);
      for (std::size_t i = 1; i < want; ++i)
        out.pool.push_back(out.compiled.create_infer_request());
    }

    // Pre-size the staging mirrors and record the output shapes NOW, at warmup,
    // so run() neither allocates nor guesses. (Staging is only USED when the
    // caller binds a space the compiled model cannot take directly.)
    for (const auto &port : out.compiled.inputs())
      if (const std::size_t n = shape_bytes(port); n)
        out.staging[port.get_any_name()].resize(n);
    for (const auto &port : out.compiled.outputs()) {
      const auto ps = port.get_partial_shape();
      if (!ps.is_static())
        continue;
      const auto sh = ps.to_shape();
      out.out_shapes[port.get_any_name()] =
          std::vector<std::int64_t>(sh.begin(), sh.end());
      out.staging[port.get_any_name()].resize(shape_bytes(port));
    }
    return true;
  }

  [[nodiscard]] static std::size_t shape_bytes(const ov::Output<const ov::Node> &port) {
    const auto ps = port.get_partial_shape();
    if (!ps.is_static())
      return 0;
    const auto sh = ps.to_shape();
    std::size_t n = port.get_element_type().size();
    for (auto d : sh)
      n *= d;
    return n;
  }

  [[nodiscard]] Variant *select(const std::vector<std::int64_t> &primary) {
    if (!primary.empty()) {
      // 1. Exact static shape — the fastest artefact when one was prebuilt.
      auto it = variants.find(shape_key(primary));
      if (it != variants.end() && it->second.key_shape == primary)
        return &it->second;

      // 2. Same shape with a DYNAMIC batch dimension.
      //
      // A per-(width,batch) artefact set is the TensorRT shape: every shape needs
      // its own profile. OpenVINO does not work that way — it handles dynamic
      // dims natively — and the per-shape set is expensive here, because each
      // CompiledModel carries its own plugin-packed weights AND, under the
      // throughput hint, its own per-stream scratch. Measured on this part: the
      // full ladder (5 widths x 7 rungs = up to 35 artefacts) costs 3.2 GB per
      // replica at the latency hint and 6.7 GB at throughput, for a 4.3 MB
      // model — which OOM-killed a 10-replica run at 26 GB RSS.
      //
      // Compiling ONE artefact per WIDTH with a dynamic batch collapses that by
      // the length of the batch ladder while keeping the width static (width is
      // what actually changes the kernel shapes). This lookup is what makes such
      // an artefact reachable: a request for [6,3,48,320] finds the [-1,3,48,320]
      // variant instead of falling through to the fully dynamic one.
      //
      // NOT counted as a shape_miss: the artefact was prebuilt deliberately and
      // no compilation happens here. shape_misses() must keep meaning "the
      // warmup matrix did not cover this", or it stops being a usable signal.
      std::vector<std::int64_t> dyn_batch = primary;
      dyn_batch[0] = -1;
      it = variants.find(shape_key(dyn_batch));
      if (it != variants.end() && it->second.key_shape == dyn_batch)
        return &it->second;
    }
    ++misses;
    return dynamic_variant.get();
  }
};

OpenVINOEngine::OpenVINOEngine(DeviceType device,
                               std::shared_ptr<L0Allocator> alloc)
    : impl_(std::make_unique<Impl>(device, std::move(alloc))) {}
OpenVINOEngine::~OpenVINOEngine() = default;

bool OpenVINOEngine::load(const std::string &model_path) {
  auto &I = *impl_;
  try {
    // On-disk compiled-blob cache: the second and later starts skip the
    // expensive graph compile entirely, which is what makes prebuilding a whole
    // (width,batch) matrix affordable at boot.
    if (const std::string cd = env::env_or("OV_CACHE_DIR", ""); !cd.empty())
      I.core.set_property(ov::cache_dir(cd));

    I.model = I.core.read_model(model_path);

    // Device-resident I/O. We take the GPU plugin's remote context and later
    // bind our USM pointers against it (create_tensor(type, shape, usm_ptr)).
    //
    // HONEST STATUS — this is the make-or-break unknown of the whole backend and
    // it CANNOT be checked without an Intel GPU:
    //   * OpenVINO's public GPU interop is context-scoped. A USM pointer is
    //     bindable only if it belongs to the same underlying context.
    //   * Whether the DPC++ runtime's sycl::context and the GPU plugin's default
    //     context are the same ze_context_handle_t is driver/version dependent.
    //   * If they are not, the two supported fixes are (a) construct the plugin
    //     context FROM our handle via ov::intel_gpu::ocl::ClContext, or (b)
    //     invert ownership and allocate through the plugin
    //     (ClContext::create_usm_device_tensor), handing SYCL those pointers.
    // Everything below is written so that a failure here is SAFE, not silent:
    // has_remote stays false, caps().io_space reports Host, and run() stages
    // through pre-sized mirrors. Correct but with an extra copy — never wrong.
    I.has_remote = false;
#if defined(TURBO_OCR_HAS_OV_USM)
    if (I.device == DeviceType::GPU && I.alloc && I.alloc->has_device() &&
        I.alloc->native_l0_context() != nullptr) {
      try {
        I.remote =
            I.core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
        I.has_remote = true;
      } catch (const std::exception &) {
        I.has_remote = false;
      }
    }
#endif

    // The always-present dynamic fallback variant.
    I.dynamic_variant = std::make_unique<Impl::Variant>();
    if (!I.build_variant({}, *I.dynamic_variant))
      return false;

    I.in_names.clear();
    I.out_names.clear();
    for (const auto &in : I.dynamic_variant->compiled.inputs())
      I.in_names.push_back(in.get_any_name());
    for (const auto &out : I.dynamic_variant->compiled.outputs())
      I.out_names.push_back(out.get_any_name());

    I.loaded = true;
    return true;
  } catch (const std::exception &) {
    // Recoverable: the backend disables this stage cleanly (ORT semantics).
    I.loaded = false;
    return false;
  }
}

std::size_t
OpenVINOEngine::prebuild(const std::vector<std::vector<std::int64_t>> &shapes) {
  auto &I = *impl_;
  if (!I.loaded)
    return 0;
  std::size_t built = 0;
  for (const auto &s : shapes) {
    if (s.empty())
      continue;
    const auto k = shape_key(s);
    if (auto it = I.variants.find(k); it != I.variants.end() && it->second.key_shape == s)
      continue;
    try {
      Impl::Variant v;
      if (I.build_variant(s, v)) {
        I.variants.emplace(k, std::move(v));
        ++built;
      }
    } catch (const std::exception &) {
      // A shape the model genuinely cannot take (e.g. NPU static-shape limits)
      // is not fatal: that shape will route through the dynamic variant and show
      // up in shape_misses().
    }
  }
  return built;
}

std::vector<std::int64_t>
OpenVINOEngine::output_shape(const std::vector<std::int64_t> &primary,
                             const std::string &out_name) const {
  const auto &I = *impl_;
  auto it = I.variants.find(shape_key(primary));
  if (it == I.variants.end() || it->second.key_shape != primary)
    return {};
  auto oit = it->second.out_shapes.find(out_name);
  return oit == it->second.out_shapes.end() ? std::vector<std::int64_t>{}
                                            : oit->second;
}

std::size_t OpenVINOEngine::shape_misses() const noexcept { return impl_->misses; }
bool OpenVINOEngine::is_loaded() const noexcept { return impl_->loaded; }

backend::EngineCaps OpenVINOEngine::caps() const {
  backend::EngineCaps c;
  c.io_space = impl_->io_space();
  c.async = false; // see the header: correctness, not oversight
  c.caller_owns_outputs = true;
  c.multi_io = true;
  c.dynamic_shapes = true;
  // The GPU plugin JIT-compiles kernels for every NEW shape it sees, on the
  // dynamic variant as much as on a fresh static compile — measured on a
  // UHD 770: det_tiny 121 ms/img in-pipeline (open per-image canvas set) vs
  // 15.4 ms at one fixed shape in benchmark_app. The NPU plugin is stricter
  // still (static-shape oriented). The CPU plugin reshapes for ~free.
  // Callers gate detection::snap_det_canvas_grid on this bit.
  c.per_shape_jit = (impl_->device != DeviceType::CPU);
  c.graph = false;
  c.has_profiles = false;
  c.thread_safe_concurrent = false; // one InferRequest per engine per thread
  c.dtypes = {backend::DType::F32, backend::DType::I64, backend::DType::I32};
  return c;
}

const std::vector<std::string> &OpenVINOEngine::input_names() const {
  return impl_->in_names;
}
const std::vector<std::string> &OpenVINOEngine::output_names() const {
  return impl_->out_names;
}

bool OpenVINOEngine::run(const std::vector<backend::DeviceTensor> &inputs,
                         const std::vector<backend::DeviceTensor> &outputs,
                         std::vector<backend::OutputLease> &leases,
                         backend::DeviceQueue &queue) {
  leases.clear(); // caller_owns_outputs == true: never a lease
  auto &I = *impl_;
  if (!I.loaded || inputs.empty())
    return false;

  auto *v = I.select(inputs[0].shape);
  if (!v)
    return false;

  // ---- BATCH-SPLIT CONCURRENCY -------------------------------------------
  //
  // The seam's contract is that run() returns with outputs valid, and that stays
  // true here. What changes is HOW the forward pass is executed: instead of one
  // synchronous infer over the whole batch, the batch is cut into chunks that
  // run on separate InferRequests concurrently, then joined before returning.
  //
  // WHY: OpenVINO's throughput figures assume several requests in flight across
  // its streams. One synchronous request gets streams=1 behaviour whatever hint
  // the model was compiled with — measured on Core Ultra 7 265T, rec_tiny does
  // 476 crops/s at streams=1 vs 2144 with the throughput hint, and the pipeline
  // was landing on 145 ms/page, exactly the streams=1 number. This is the gap.
  //
  // Splitting along BATCH is safe because every tensor here is batch-major and
  // batch rows are INDEPENDENT: chunk c owns the contiguous byte range
  // [lo*row_bytes, hi*row_bytes) of each input and output. No tensor is
  // reinterpreted, no result is combined — this is not a numerical change and
  // the golden diffs must be bit-identical.
  //
  // Guarded to the cases where those assumptions provably hold:
  //   * one input (multi-input models like layout are not batch-parallel here),
  //   * every output caller-owned (an engine-owned lease has no per-chunk
  //     destination to write into),
  //   * a DYNAMIC-batch variant, so each chunk's smaller batch is a shape the
  //     compiled model actually accepts,
  //   * no staging (bound space matches), so slicing is a pointer offset,
  //   * batch > 1 and a pool to use.
  // Anything else falls through to the original single-request path untouched.
  const auto space_pre = I.io_space();
  const bool splittable =
      inputs.size() == 1 && !v->pool.empty() && !v->key_shape.empty() &&
      v->key_shape[0] == -1 && !inputs[0].shape.empty() && inputs[0].shape[0] > 1 &&
      inputs[0].space == space_pre &&
      std::all_of(outputs.begin(), outputs.end(), [&](const backend::DeviceTensor &o) {
        return o.data != nullptr && o.space == space_pre && !o.shape.empty() &&
               o.shape[0] == inputs[0].shape[0];
      });

  if (splittable) {
    try {
      queue.synchronize();
      const std::int64_t B = inputs[0].shape[0];
      const std::size_t nreq = std::min<std::size_t>(v->pool.size() + 1,
                                                     static_cast<std::size_t>(B));
      // Even chunks, remainder spread over the first few, so no request idles on
      // an empty slice and the largest chunk is at most one row bigger.
      const std::int64_t base = B / static_cast<std::int64_t>(nreq);
      const std::int64_t rem = B % static_cast<std::int64_t>(nreq);

      auto row_bytes = [](const backend::DeviceTensor &t) {
        std::size_t n = 1;
        for (std::size_t i = 1; i < t.shape.size(); ++i)
          n *= static_cast<std::size_t>(t.shape[i] > 0 ? t.shape[i] : 0);
        return n * dtype_size(t.dtype);
      };

      std::vector<ov::InferRequest *> used;
      used.reserve(nreq);
      std::int64_t lo = 0;
      for (std::size_t c = 0; c < nreq; ++c) {
        const std::int64_t rows = base + (static_cast<std::int64_t>(c) < rem ? 1 : 0);
        if (rows <= 0) continue;
        ov::InferRequest *req = (c == 0) ? &v->request : &v->pool[c - 1];

        auto bind_slice = [&](const backend::DeviceTensor &t) {
          std::vector<std::size_t> sh(t.shape.begin(), t.shape.end());
          sh[0] = static_cast<std::size_t>(rows);
          auto *p = static_cast<char *>(t.data) +
                    static_cast<std::size_t>(lo) * row_bytes(t);
          req->set_tensor(t.name, ov::Tensor(to_ov(t.dtype), ov::Shape(sh), p));
        };
        bind_slice(inputs[0]);
        for (const auto &o : outputs) bind_slice(o);

        req->start_async();
        used.push_back(req);
        lo += rows;
      }
      for (auto *req : used) req->wait();
      leases.clear(); // caller-owned outputs only on this path, by construction
      return true;
    } catch (const std::exception &) {
      // Any binding the plugin refuses falls back to the single-request path
      // below rather than failing the stage.
    }
  }

  try {
    const auto space = I.io_space();

    // (a) Make sure everything the SYCL lane wrote (warp_crops, resize) has
    //     landed before OpenVINO reads it. One barrier per forward pass; the
    //     honest cost of not sharing a command queue (README item 2).
    queue.synchronize();

    // (b) Bind. Matching space => zero-copy in place. Mismatched space => stage
    //     through the variant's pre-allocated mirror (no hot-path alloc unless
    //     the shape was never prebuilt, which shape_misses() surfaces).
    auto bind = [&](const backend::DeviceTensor &t, bool is_input) {
      const ov::Shape shape(t.shape.begin(), t.shape.end());
      const auto et = to_ov(t.dtype);
      const std::size_t bytes = elem_count(t.shape) * dtype_size(t.dtype);

      if (t.space == space) {
#if defined(TURBO_OCR_HAS_OV_USM)
        if (space == backend::DeviceKind::L0) {
          // USM device pointer -> USMTensor (SharedMemType::USM_USER_BUFFER).
          // The plugin accepts a USM pointer allocated on ITS context; whether
          // SYCL's context IS that context is bring-up item 1 and is
          // UNVALIDATED here. If it is not, create_tensor throws, run() returns
          // false and the stage reports failure — it never silently reads
          // foreign memory.
          v->request.set_tensor(t.name, I.remote.create_tensor(et, shape, t.data));
          return;
        }
#endif
        v->request.set_tensor(t.name, ov::Tensor(et, shape, t.data));
        return;
      }

      // Degraded staging path.
      auto &buf = v->staging[t.name];
      if (buf.size() < bytes)
        buf.resize(bytes); // only for non-prebuilt shapes; see shape_misses()
      if (is_input) {
        if (t.space == backend::DeviceKind::Host)
          std::memcpy(buf.data(), t.data, bytes);
        else if (I.alloc)
          I.alloc->copy_d2h(buf.data(), t.data, bytes, queue);
        queue.synchronize();
      }
      v->request.set_tensor(t.name, ov::Tensor(et, shape, buf.data()));
    };

    for (const auto &in : inputs)
      bind(in, /*is_input=*/true);
    for (const auto &out : outputs)
      if (out.data != nullptr) // data==nullptr => engine-owned, leased back
        bind(out, /*is_input=*/false);

    // (c) Synchronous forward pass — outputs are valid on return.
    v->request.infer();

    // (d) Copy staged outputs back into the caller's buffers, and lease the
    //     engine-owned ones. Outputs the caller never mentioned stay inside
    //     OpenVINO and are never touched (layout's mask tensor).
    for (const auto &out : outputs) {
      if (out.data == nullptr) {
        const ov::Tensor t = v->request.get_tensor(out.name);
        const auto sh = t.get_shape();
        backend::OutputLease lease;
        lease.name = out.name;
        lease.dtype = out.dtype;
        lease.shape.assign(sh.begin(), sh.end());
        // Host-visible view. A remote (device) tensor cannot be dereferenced on
        // the host, so it is copied into the variant's pre-sized mirror; these
        // leased outputs are the small, data-dependent ones by construction.
        if (t.is<ov::RemoteTensor>()) {
          auto &buf = v->staging[out.name];
          const std::size_t bytes = t.get_byte_size();
          if (buf.size() < bytes)
            buf.resize(bytes);
          ov::Tensor host_view(t.get_element_type(), sh, buf.data());
          t.as<ov::RemoteTensor>().copy_to(host_view);
          lease.data = buf.data();
        } else {
          lease.data = t.data();
        }
        lease.space = backend::DeviceKind::Host;
        leases.push_back(std::move(lease));
        continue;
      }
      if (out.space == space)
        continue;
      auto it = v->staging.find(out.name);
      if (it == v->staging.end())
        continue;
      const std::size_t bytes = elem_count(out.shape) * dtype_size(out.dtype);
      if (out.space == backend::DeviceKind::Host)
        std::memcpy(out.data, it->second.data(), bytes);
      else if (I.alloc)
        I.alloc->copy_h2d(out.data, it->second.data(), bytes, queue);
    }
    queue.synchronize();
    return true;
  } catch (const std::exception &) {
    return false; // recoverable, like ORT
  }
}

#else // !TURBO_OCR_HAS_OPENVINO ------------------------------------------------

struct OpenVINOEngine::Impl {
  DeviceType device;
  std::shared_ptr<L0Allocator> alloc;
  std::vector<std::string> in_names, out_names;
  std::size_t misses = 0;
  Impl(DeviceType d, std::shared_ptr<L0Allocator> a)
      : device(d), alloc(std::move(a)) {}
};

OpenVINOEngine::OpenVINOEngine(DeviceType device,
                               std::shared_ptr<L0Allocator> alloc)
    : impl_(std::make_unique<Impl>(device, std::move(alloc))) {}
OpenVINOEngine::~OpenVINOEngine() = default;

// Fails loudly-but-recoverably: the backend reports the stage unavailable
// instead of silently producing empty OCR.
bool OpenVINOEngine::load(const std::string &) { return false; }
std::size_t OpenVINOEngine::prebuild(const std::vector<std::vector<std::int64_t>> &) {
  return 0;
}
std::vector<std::int64_t>
OpenVINOEngine::output_shape(const std::vector<std::int64_t> &,
                             const std::string &) const {
  return {};
}
std::size_t OpenVINOEngine::shape_misses() const noexcept { return impl_->misses; }
bool OpenVINOEngine::is_loaded() const noexcept { return false; }

backend::EngineCaps OpenVINOEngine::caps() const {
  backend::EngineCaps c;
  // Without OpenVINO there is no remote context, so I/O is host space.
  c.io_space = backend::DeviceKind::Host;
  c.async = false;
  c.caller_owns_outputs = true;
  c.multi_io = true;
  c.dynamic_shapes = true;
  c.graph = false;
  c.has_profiles = false;
  c.thread_safe_concurrent = false;
  c.dtypes = {backend::DType::F32, backend::DType::I64, backend::DType::I32};
  return c;
}
const std::vector<std::string> &OpenVINOEngine::input_names() const {
  return impl_->in_names;
}
const std::vector<std::string> &OpenVINOEngine::output_names() const {
  return impl_->out_names;
}
bool OpenVINOEngine::run(const std::vector<backend::DeviceTensor> &,
                         const std::vector<backend::DeviceTensor> &,
                         std::vector<backend::OutputLease> &leases,
                         backend::DeviceQueue &) {
  leases.clear();
  return false;
}

#endif

} // namespace turbo_ocr::intel
