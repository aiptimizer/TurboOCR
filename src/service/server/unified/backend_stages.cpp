// stages.cpp — see stages.h. The ONE stage bootstrap for every backend.

#include "turbo_ocr/service/server/unified/backend_stages.h"

#include <filesystem>
#include <stdexcept>
#include <utility>

#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"

namespace turbo_ocr::server {
namespace {

// Translate the resolved ServerConfig into the seam's BackendConfig. Every
// backend reads its finer per-model knobs from env (exactly as stages_gpu /
// stages_cpu do today); this carries only what the server resolves generically.
backend::BackendConfig to_backend_config(const ServerConfig &cfg,
                                         bool want_tables, bool want_formulas) {
  backend::BackendConfig bc;
  bc.det_model = cfg.det_onnx;
  bc.rec_model = cfg.rec_paths.rec;
  bc.rec_dict = cfg.rec_paths.dict;
  bc.cls_model = cfg.disable_angle_cls ? std::string{} : cfg.cls_onnx;
  bc.layout_model = cfg.layout_disabled ? std::string{} : cfg.layout_onnx;
  // Autorotate is opt-in on the presence of the model file: a missing
  // doc_ori.onnx soft-disables the stage (never fatal), matching both mains.
  bc.doc_orient_model =
      std::filesystem::exists(cfg.doc_ori_onnx) ? cfg.doc_ori_onnx : std::string{};
  bc.pool_size = cfg.pipeline_pool_size.value_or(0);
  // Engine mode: --engine-mode, else TURBO_ENGINE_MODE, else Auto.
  std::string mode = cfg.engine_mode;
  if (mode.empty()) mode = env::env_or("TURBO_ENGINE_MODE", "");
  bc.mode = backend::parse_engine_mode(mode);
  // Provider overrides for the fast path (the backend fills in its own default
  // provider when this stays empty).
  //
  // TURBO_EP_DEVICE is ONE knob for every vendor ON THE ONNX/FAST PATH, because
  // device selection is generic policy there: an OpenVINO device NAME
  // (AUTO|CPU|GPU|NPU|GPU.1) for the intel arm, or a bare ordinal ("1") for the
  // CUDA/ROCm/DML arms, both resolved in src/onnx/ep_options.h
  // (openvino_options() for the name, device_id_for() for the ordinal). It used
  // to reach only the OpenVINO appender, so `TURBO_EP_DEVICE=1 --backend nvidia
  // --engine-mode onnx` ran on GPU 0 with no diagnostic.
  //
  // It stops at the seam, though: ep travels no further than
  // cpu::make_onnx_stages(), so the NATIVE/ultra engines never see it and each
  // still picks its device its own way (intel: OV_DEVICE, read by the backend
  // factory; amd: the ordinal handed to make_rocm_backend(), default 0; nvidia
  // TensorRT and apple MPSGraph: no device selector at all). Setting it on a
  // native run is silently ignored — unifying that means threading the device
  // through Backend construction, not this line.
  bc.ep.device = env::env_or("TURBO_EP_DEVICE", "");
  bc.ep.fp16 = env::env_or("TURBO_EP_FP16", "1") != "0";
  bc.want_layout = !cfg.layout_disabled;
  bc.want_tables = want_tables;
  bc.want_formulas = want_formulas;
  return bc;
}

} // namespace

BackendRuntime build_backend_runtime(std::string_view backend_name,
                                     const ServerConfig &cfg) {
  // Install the per-model detection base BEFORE any stage loads: every
  // backend's detector resolves its config through the no-arg
  // read_det_resize()/read_db_params(), whose defaults come from this base
  // (det_config.h). Without this line the registry's det_cfg — tiny's
  // box_thresh 0.40 — reached ServerConfig and was then read by nothing, so
  // every tier ran the 0.45 default.
  detection::set_det_config_base(cfg.det_cfg.resize, cfg.det_cfg.db);
  BackendRuntime rt;
  rt.backend = backend::make_backend(backend_name);
  if (!rt.backend) {
    std::string avail;
    for (auto n : backend::available_backends()) {
      if (!avail.empty()) avail += ", ";
      avail.append(n);
    }
    throw std::runtime_error("backend '" + std::string(backend_name) +
                             "' is not compiled into this build (available: " +
                             avail + ")");
  }
  // PROVISIONAL caps — only pool sizing below may use these. Several backends
  // do not know their own device/async/mode until load_stages() has run (that
  // is where each vendor resolves native-vs-onnx from the artefacts actually
  // on disk), so the values here can describe a path the server never took.
  // Refreshed after the pool is built; see the re-read below.
  rt.caps = rt.backend->caps();

  // Which optional modalities the operator routed. Reading the routing table
  // here (instead of per-backend env sniffing) is what lets ONE bootstrap serve
  // every vendor: the table is device-neutral by construction.
  bool want_tables = false, want_formulas = false;
  try {
    const backend_routing::RoutingTable routing =
        backend_routing::load_routing_config();
    want_tables = backend_routing::resolve(routing, "table") != nullptr;
    want_formulas = backend_routing::resolve(routing, "formula") != nullptr;
  } catch (const std::exception &e) {
    // Invalid routing config is fatal (same fail-fast posture as ServerConfig).
    throw std::runtime_error(std::string("routing config invalid: ") + e.what());
  }

  rt.pool_size = cfg.pipeline_pool_size.value_or(rt.caps.recommended_pool_size);
  if (rt.pool_size < 1) rt.pool_size = 1;

  const backend::BackendConfig bc =
      to_backend_config(cfg, want_tables, want_formulas);

  TOCR_LOG_INFO("Backend selected", "backend", rt.caps.name, "device",
                backend::device_kind_name(rt.caps.device), "pool_size",
                rt.pool_size, "async", rt.caps.async);

  std::vector<pipeline::UnifiedPipelineEntry> entries;
  entries.reserve(static_cast<std::size_t>(rt.pool_size));
  for (int i = 0; i < rt.pool_size; ++i) {
    backend::StageSet stages = rt.backend->load_stages(bc);
    if (!stages.available.detector || !stages.available.recognizer)
      throw std::runtime_error(
          "backend load_stages() did not produce the required detector + "
          "recognizer stages — refusing to start");
    const bool layout_ok =
        stages.available.optional.get(capability::CapabilityId::Layout);

    auto queue = rt.backend->make_queue();
    auto pipe = std::make_unique<pipeline::UnifiedOcrPipeline>(
        *rt.backend, std::move(stages), std::move(queue));

    // Optional stage bootstrap. These return false ONLY when an explicitly
    // configured LOCAL backend failed to load (a remote endpoint that is merely
    // unreachable at boot registers and retries per request).
    if (!pipe->load_router_models())
      throw std::runtime_error(
          "formula backend failed to load — refusing to start");
    if (!pipe->load_table_backend())
      throw std::runtime_error(
          "table backend failed to load — refusing to start");
    pipe->warmup();

    // Availability is read from what ACTUALLY loaded into the pipeline (single
    // source of truth for the tables=1/formulas=1 fail-loud gate), never from
    // operator intent. All entries load identically, so the last wins.
    rt.available.set(capability::CapabilityId::Layout, layout_ok);
    rt.available.set(capability::CapabilityId::Table,
                     pipe->has_default_table_backend());
    rt.available.set(capability::CapabilityId::Formula,
                     pipe->has_default_formula_backend());
    rt.available.set(capability::CapabilityId::DocOrientation,
                     pipe->has_doc_ori());

    entries.push_back(pipeline::UnifiedPipelineEntry{std::move(pipe), nullptr});
  }

  // AUTHORITATIVE caps, re-read now that load_stages() has resolved the engine
  // mode. This is not cosmetic: a backend that asked for Auto and fell back to
  // the onnx path reports a DIFFERENT device and async flag once it knows (the
  // Apple onnx path runs host stages on a synchronous queue, not Metal+async).
  // Three things downstream consume this and were all being handed the
  // pre-resolution guess — /capabilities and the startup log (which would name
  // a device the server is not using), the io-thread sizing in server_main, and
  // register_device_readback(), which keys the remote-VLM readback hook on
  // caps.device and would have filed it under the wrong DeviceKind.
  rt.caps = rt.backend->caps();

  // TURBO_EP_DEVICE only reaches the ONNX/fast path (it is bound into EpConfig
  // and consumed at make_onnx_stages); the native engines each pick their device
  // their own way — intel via OV_DEVICE, amd via a constructor ordinal, nvidia
  // TensorRT and apple MPSGraph not at all. So on a native run the knob is
  // silently ignored, which is exactly the kind of set-knob-does-nothing the
  // observability rule exists to prevent. Now that the re-read above has
  // resolved the real mode, say so once instead of leaving it silent.
  if (rt.caps.mode == backend::EngineMode::Native &&
      env::env_present("TURBO_EP_DEVICE")) {
    TOCR_LOG_WARN(
        "TURBO_EP_DEVICE is set but this backend resolved to the NATIVE engine, "
        "which does not read it — the device selector applies only to the ONNX "
        "path (TURBO_ENGINE_MODE=onnx). Ignored.",
        "backend", rt.caps.name, "device", env::env_or("TURBO_EP_DEVICE", ""));
  }

  rt.pool = std::make_shared<pipeline::UnifiedPipelinePool>(std::move(entries));
  // Logged from the capability table, so a capability added there shows up in
  // the startup log without anyone remembering to extend this line.
  {
    std::string loaded_csv;
    for (const auto &cap : capability::kCapabilities) {
      if (!loaded_csv.empty()) loaded_csv += ' ';
      loaded_csv += std::string(cap.name) + '=' +
                    (rt.available.get(cap.id) ? '1' : '0');
    }
    TOCR_LOG_INFO("Stages loaded", "capabilities",
                  std::string_view(loaded_csv));
  }
  return rt;
}

} // namespace turbo_ocr::server
