#include "amd/engine/migraphx_engine.h"

#include "amd/queue/hip_queue.h"

#include "turbo_ocr/base/env_utils.h"

#include <cstdio>
#include <filesystem>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <unordered_map>

// MIGraphX C++ API. Only this TU sees it (pImpl), so stage/backend TUs stay
// migraphx-free.
#include <migraphx/migraphx.hpp>

namespace turbo_ocr::amd {

namespace {

migraphx_shape_datatype_t to_mgx_dtype(backend::DType dt) {
  switch (dt) {
  case backend::DType::F32: return migraphx_shape_float_type;
  case backend::DType::F16: return migraphx_shape_half_type;
  case backend::DType::I64: return migraphx_shape_int64_type;
  case backend::DType::I32: return migraphx_shape_int32_type;
  case backend::DType::U8:  return migraphx_shape_uint8_type;
  }
  return migraphx_shape_float_type;
}

backend::DType from_mgx_dtype(migraphx_shape_datatype_t t) {
  switch (t) {
  case migraphx_shape_float_type:  return backend::DType::F32;
  case migraphx_shape_half_type:   return backend::DType::F16;
  case migraphx_shape_int64_type:  return backend::DType::I64;
  case migraphx_shape_int32_type:  return backend::DType::I32;
  case migraphx_shape_uint8_type:  return backend::DType::U8;
  default:                         return backend::DType::F32;
  }
}

// A stable, cheap key for a set of input shapes. Built by appending dims into a
// small string — no allocation beyond the string itself, and comparing whole
// keys avoids the per-dim loop of a vector-of-vectors map. Callers build it once
// per run() and hash-lookup with it.
std::string shape_key(const std::vector<std::vector<std::int64_t>> &dims) {
  std::string k;
  k.reserve(dims.size() * 12);
  for (const auto &d : dims) {
    for (auto v : d) {
      k += std::to_string(v);
      k += ',';
    }
    k += ';';
  }
  return k;
}

namespace fs = std::filesystem;

// Same resolution order as the TRT engine cache (trt_engine_cache.cpp): env
// override, then ~/.cache/turbo-ocr (shared dir; .mxr files coexist with .trt
// by extension), then /tmp. Empty return = caching disabled.
std::string mgx_cache_dir() {
  if (std::string dir = env::env_or("MIGRAPHX_ENGINE_CACHE", ""); !dir.empty()) {
    if (dir == "off" || dir == "0")
      return {};
    std::error_code ec;
    fs::create_directories(dir, ec);
    return ec ? std::string{} : dir;
  }
  // HOME read raw on purpose: ambient identity, not a TurboOCR knob (same
  // rationale as the TRT cache).
  if (auto *home = std::getenv("HOME")) {  // pre-commit-allow-getenv (not a knob)
    auto dir = std::string(home) + "/.cache/turbo-ocr";
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (!ec)
      return dir;
  }
  auto dir = std::string("/tmp/turbo-ocr-engines");
  std::error_code ec;
  fs::create_directories(dir, ec);
  return ec ? std::string{} : dir;
}

} // namespace

struct MIGraphXEngine::Impl {
  int device_id = 0;
  bool fp16 = false;
  bool loaded = false;
  std::string model_path;

  std::vector<std::string> in_names;
  std::vector<std::string> out_names;

  // ONE compiled program per input-shape signature. MIGraphX compiles for
  // concrete shapes, so this map IS the "executables cached by (width,batch)"
  // requirement of the performance gate — every entry is a compile that run()
  // does not pay.
  struct Variant {
    migraphx::program prog;
    migraphx::program_parameter_shapes param_shapes;
  };
  std::unordered_map<std::string, Variant> variants;
  const Variant *last_used = nullptr; // fast path: same shape as last call
  std::string last_key;

  // Engine-owned outputs from the last eval, kept alive so the OutputLease
  // device pointers remain valid until the next run() overwrites them.
  // optional because migraphx::arguments has no default constructor (it is a
  // handle type only ever produced by an eval) — real-header fact, the shim
  // originally permitted default construction.
  std::optional<migraphx::arguments> last_outputs;

  std::size_t hot_compiles = 0;

  // Per-gfx persistent compile cache. A warm start is 42 graph compiles (35 rec
  // + 6 cls + 1 layout); with the cache each is paid once per (model, shape,
  // gfx, fp16, ROCm) and every later start is a migraphx::load(). "" = disabled
  // (cache dir unavailable or device properties unreadable).
  std::string cache_dir;   // resolved once in ensure_cache_identity()
  std::string arch_name;   // hipDeviceProp_t::gcnArchName, e.g. "gfx942"
  std::string ver_tag;     // hip driver+runtime versions baked into the key
  bool cache_identity_done = false;

  void ensure_cache_identity() {
    if (cache_identity_done)
      return;
    cache_identity_done = true;
    cache_dir = mgx_cache_dir();
    if (cache_dir.empty())
      return;
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device_id) != hipSuccess) {
      // No usable device identity => a cached program's target is unknowable;
      // disable rather than risk loading a program compiled for another gfx.
      cache_dir.clear();
      return;
    }
    arch_name = prop.gcnArchName;
    int drv = 0, rt = 0;
    (void)hipDriverGetVersion(&drv);
    (void)hipRuntimeGetVersion(&rt);
    ver_tag = ":drv" + std::to_string(drv) + ":rt" + std::to_string(rt);
  }

  // Mirrors the TRT cache key discipline (trt_engine_cache.cpp): model identity
  // by path+size+mtime, plus everything that changes the compiled artifact —
  // the shape signature, fp16, the gfx target, and the ROCm driver/runtime.
  std::string cache_path_for(const std::string &key) {
    ensure_cache_identity();
    if (cache_dir.empty())
      return {};
    std::error_code ec;
    const auto size = fs::file_size(model_path, ec);
    if (ec)
      return {};
    const auto mtime = static_cast<long long>(
        fs::last_write_time(model_path, ec).time_since_epoch().count());
    if (ec)
      return {};
    const std::string full = model_path + ":" + std::to_string(size) + ":" +
                             std::to_string(mtime) + ":" + key +
                             (fp16 ? ":fp16" : ":fp32") + ":" + arch_name +
                             ver_tag;
    const auto hash = std::hash<std::string>{}(full);
    return cache_dir + "/mgx_" + fs::path(model_path).stem().string() + "_" +
           std::to_string(hash) + ".mxr";
  }

  // Parse + compile ONE variant. `dims` empty => use the model's declared input
  // shapes verbatim (what load() does). Tries the persistent .mxr cache first;
  // on a compile, saves the result back (atomic tmp+rename so a crashed writer
  // never leaves a torn file for the next reader).
  Variant *compile_variant(const std::string &key,
                           const std::vector<std::vector<std::int64_t>> &dims) {
    const std::string mxr = cache_path_for(key);
    if (!mxr.empty()) {
      std::error_code ec;
      if (fs::exists(mxr, ec) && !ec) {
        try {
          migraphx::program prog = migraphx::load(mxr.c_str());
          Variant v;
          v.param_shapes = prog.get_parameter_shapes();
          v.prog = std::move(prog);
          auto [it, _] = variants.insert_or_assign(key, std::move(v));
          return &it->second;
        } catch (const std::exception &e) {
          // Corrupt or version-skewed artifact: drop it and recompile.
          std::fprintf(stderr,
                       "[MIGraphXEngine] stale cache %s (%s) — recompiling\n",
                       mxr.c_str(), e.what());
          fs::remove(mxr, ec);
        }
      }
    }
    try {
      migraphx::onnx_options onnx_opts;
      if (!dims.empty()) {
        // Pin every input parameter to a concrete shape BEFORE parsing, so the
        // parser materializes a static graph for exactly this (batch, width).
        for (std::size_t i = 0; i < dims.size() && i < in_names.size(); ++i) {
          std::vector<std::size_t> d(dims[i].begin(), dims[i].end());
          onnx_opts.set_input_parameter_shape(in_names[i], d);
        }
      }
      migraphx::program prog =
          migraphx::parse_onnx(model_path.c_str(), onnx_opts);

      if (fp16)
        migraphx::quantize_fp16(prog);

      migraphx::target t = migraphx::target("gpu");
      migraphx::compile_options copts;
      // offload_copy=false => the program consumes/produces DEVICE buffers; the
      // caller supplies device input pointers and MIGraphX returns device output
      // arguments. This is the device-resident contract; with offload_copy=true
      // MIGraphX would silently H2D/D2H every call.
      copts.set_offload_copy(false);
      prog.compile(t, copts);

      if (!mxr.empty()) {
        // Atomic publish: save to a pid-suffixed temp in the SAME directory,
        // then rename over the final name. Two replicas warming concurrently
        // race benignly — last rename wins with identical content.
        const std::string tmp = mxr + ".tmp." + std::to_string(::getpid());
        try {
          migraphx::save(prog, tmp.c_str());
          std::error_code ec;
          fs::rename(tmp, mxr, ec);
          if (ec)
            fs::remove(tmp, ec);
        } catch (const std::exception &e) {
          std::fprintf(stderr, "[MIGraphXEngine] cache save %s failed: %s\n",
                       mxr.c_str(), e.what());
          std::error_code ec;
          fs::remove(tmp, ec);
        }
      }

      Variant v;
      v.param_shapes = prog.get_parameter_shapes();
      v.prog = std::move(prog);
      auto [it, _] = variants.insert_or_assign(key, std::move(v));
      return &it->second;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "[MIGraphXEngine] compile(%s, shapes=%s) failed: %s\n",
                   model_path.c_str(), key.c_str(), e.what());
      return nullptr;
    }
  }
};

MIGraphXEngine::MIGraphXEngine(int device_id) : p_(std::make_unique<Impl>()) {
  p_->device_id = device_id;
}

MIGraphXEngine::~MIGraphXEngine() = default;

void MIGraphXEngine::set_fp16(bool on) noexcept { p_->fp16 = on; }

std::size_t MIGraphXEngine::hot_path_compiles() const noexcept {
  return p_->hot_compiles;
}

bool MIGraphXEngine::load(const std::string &model_path) {
  p_->model_path = model_path;
  p_->variants.clear();
  p_->last_used = nullptr;
  p_->last_key.clear();

  // Compile the model's DECLARED shape first; this is also how we discover the
  // I/O names that warmup() and run() bind against.
  Impl::Variant *v = p_->compile_variant(/*key=*/"", /*dims=*/{});
  if (!v) {
    p_->loaded = false;
    return false;
  }

  p_->in_names.clear();
  for (auto &&name : v->param_shapes.names())
    p_->in_names.emplace_back(name);

  // Outputs are exposed positionally by MIGraphX; synthesize stable names.
  p_->out_names.clear();
  auto out_shapes = v->prog.get_output_shapes();
  for (std::size_t i = 0; i < out_shapes.size(); ++i)
    p_->out_names.emplace_back("output_" + std::to_string(i));

  p_->loaded = true;
  return true;
}

std::size_t
MIGraphXEngine::warmup(const std::vector<ShapeVariant> &variants) {
  if (!p_->loaded)
    return 0;
  std::size_t ok = 0;
  for (const auto &v : variants) {
    const std::string key = shape_key(v.input_dims);
    if (p_->variants.count(key)) {
      ++ok;
      continue;
    }
    if (p_->compile_variant(key, v.input_dims))
      ++ok;
  }
  return ok;
}

EngineCaps MIGraphXEngine::caps() const {
  EngineCaps c;
  c.io_space = backend::DeviceKind::Hip;
  c.async = true;
  c.caller_owns_outputs = false; // results via OutputLease (engine-owned device)
  c.multi_io = true;
  // True from the CALLER's point of view: run() accepts any shape. Internally
  // each shape is a separately compiled static program, cached by warmup().
  c.dynamic_shapes = true;
  c.graph = false;
  c.has_profiles = false;
  c.thread_safe_concurrent = false; // one program/context per thread (like TRT)
  c.dtypes = {backend::DType::F32, backend::DType::F16, backend::DType::I64,
              backend::DType::I32, backend::DType::U8};
  return c;
}

const std::vector<std::string> &MIGraphXEngine::input_names() const {
  return p_->in_names;
}
const std::vector<std::string> &MIGraphXEngine::output_names() const {
  return p_->out_names;
}

bool MIGraphXEngine::run(const std::vector<DeviceTensor> &inputs,
                         const std::vector<DeviceTensor> &outputs,
                         std::vector<OutputLease> &leases, DeviceQueue &queue) {
  (void)outputs; // caller_owns_outputs == false: outputs come back via leases
  if (!p_->loaded)
    return false;

  // --- Select the compiled program for this shape (hot path: a map hit) ------
  std::vector<std::vector<std::int64_t>> dims;
  dims.reserve(inputs.size());
  for (const auto &t : inputs)
    dims.push_back(t.shape);
  const std::string key = shape_key(dims);

  const Impl::Variant *var = nullptr;
  if (p_->last_used && key == p_->last_key) {
    var = p_->last_used; // same shape as the previous call — no lookup at all
  } else {
    auto it = p_->variants.find(key);
    if (it != p_->variants.end()) {
      var = &it->second;
    } else {
      // MISS. Compiling here is CORRECT but SLOW (a full graph compile inside a
      // request). Shout about it: the fix is always to add this shape to the
      // stage's warmup ladder, never to accept the stall.
      ++p_->hot_compiles;
      std::fprintf(stderr,
                   "[MIGraphXEngine] HOT-PATH COMPILE for shape [%s] on %s — "
                   "this shape is missing from the warmup ladder (compile #%zu). "
                   "Add it to the stage's warmup() call.\n",
                   key.c_str(), p_->model_path.c_str(), p_->hot_compiles);
      var = p_->compile_variant(key, dims);
      if (!var)
        return false;
    }
    p_->last_used = var;
    p_->last_key = key;
  }

  try {
    // Bind each caller device pointer as a MIGraphX argument over the compiled
    // parameter shape (zero-copy: argument views the caller's hipMalloc buffer).
    migraphx::program_parameters pp;
    for (const auto &t : inputs) {
      if (t.space != backend::DeviceKind::Hip) {
        std::fprintf(stderr, "[MIGraphXEngine] input '%s' not in Hip space\n",
                     t.name.c_str());
        return false;
      }
      // The compiled parameter shape fixes dtype + lengths; wrap the caller's
      // device data with no copy.
      migraphx::shape s = var->param_shapes[t.name.c_str()];
      if (s.type() != to_mgx_dtype(t.dtype)) {
        std::fprintf(stderr,
                     "[MIGraphXEngine] input '%s' dtype mismatch vs compiled "
                     "program\n",
                     t.name.c_str());
        return false;
      }
      pp.add(t.name.c_str(), migraphx::argument(s, t.data));
    }

    // Enqueue on the caller's hipStream so pre/forward/post stay on one lane.
    // run_async does NOT block; the caller syncs the DeviceQueue before reading.
    // The real API is a template over the stream's pointee type — it stringifies
    // the type name itself, so the hipStream_t is passed directly.
    hipStream_t stream = hip_stream_of(queue);
    p_->last_outputs = var->prog.run_async(pp, stream);

    // Surface the engine-owned device outputs as leases. Their device pointers
    // are valid until the next run() overwrites p_->last_outputs.
    leases.clear();
    leases.reserve(p_->last_outputs->size());
    for (std::size_t i = 0; i < p_->last_outputs->size(); ++i) {
      migraphx::argument arg = (*p_->last_outputs)[i];
      migraphx::shape s = arg.get_shape();
      OutputLease lease;
      lease.name = (i < p_->out_names.size()) ? p_->out_names[i]
                                              : ("output_" + std::to_string(i));
      lease.data = arg.data();
      lease.space = backend::DeviceKind::Hip;
      lease.dtype = from_mgx_dtype(s.type());
      auto lens = s.lengths();
      lease.shape.assign(lens.begin(), lens.end());
      leases.push_back(std::move(lease));
    }
    return true;
  } catch (const std::exception &e) {
    // Recoverable: log + false (mirrors ORT's caught-exception contract, unlike
    // TRT's fail-fast abort). A poisoned device is caught later by HIP_CHECK in
    // the kernel path.
    std::fprintf(stderr, "[MIGraphXEngine] run failed: %s\n", e.what());
    return false;
  }
}

} // namespace turbo_ocr::amd
