#pragma once

#include <string>

#include "turbo_ocr/service/server/bootstrap/server_config.h"
#include "turbo_ocr/core/capability.h"
#include "turbo_ocr/core/service_fns.h"
#include "turbo_ocr/service/server/work_pool.h"

namespace turbo_ocr::routes {

/// Inputs for the GET /capabilities endpoint (M6 cross-build parity).
/// Each binary fills this from its already-resolved ServerConfig + the
/// runtime-availability bools it computed at startup, then hands it to
/// register_capabilities_route. Kept as an explicit POD (rather than the
/// whole ServerConfig) so the contract is exactly the advertised surface
/// and adding a field here can't silently leak an unrelated config value.
///
/// `/capabilities` is purely additive: it lets a client discover, without
/// trial requests, which features the deployment it reached actually honors.
/// It changes no existing default, response shape, or endpoint behavior.
///
/// It used to advertise "the known GPU/CPU build divergences, which the two
/// binaries deliberately keep". There is one binary now, and one route set;
/// what still varies between deployments is which MODELS loaded and which
/// DEVICE came up, and those are `loaded` / `implemented` and
/// `backend_name` / `device_name` below.
struct CapabilitiesInfo {
  // Serialized as `"build": "gpu" | "cpu"`. It is a RUNTIME fact, not a build
  // one: true whenever the backend that came up is on an accelerator (Metal,
  // CUDA, HIP, L0), false on the host backend. The wire name predates the
  // single binary and is kept because clients read it; `device_name` below is
  // the field to use for anything new.
  bool is_gpu = false;

  // ---- features ----
  // LOADED (capability/capability.h). Serialized by iterating the capability
  // table, so a capability added there is advertised here automatically — it is
  // structurally impossible to make a capability requestable but leave it out
  // of /capabilities, which is how a client's only discovery mechanism used to
  // go stale.
  capability::CapabilityMask loaded;

  // IMPLEMENTED (capability/capability.h): what this backend+mode COULD do with
  // the right models. Advertised alongside `loaded` so an operator can tell
  // "supported here, but no model configured" (their problem to fix) from "this
  // backend cannot do it at all" (not fixable by configuration). A single flat
  // bool conflates the two and leaves them with no idea which knob to reach for.
  capability::CapabilityMask implemented = capability::CapabilityMask::all();

  // Which backend actually came up ("apple", "cpu", ...) and its device.
  // These used to live on a separate GET /capabilities/backend endpoint purely
  // because this header was frozen at the time; it no longer is.
  std::string backend_name;
  std::string device_name;
  // Which engine path the backend actually came up on, and what it could offer.
  // BackendCaps carried these three from the start with the stated purpose that
  // "/capabilities and the Python info() can never disagree with reality" — and
  // then nothing serialized them, so an Auto run that fell back from native to
  // onnx said nothing at all. They are on the wire now.
  std::string engine_mode = "onnx";
  bool has_native_engine = false;
  bool has_onnx_engine = true;

  bool profile_endpoint = false;      // GET /profile registered
  // NOTE (removed): a `gpu_routes` flag sat here, gating /ocr/markdown,
  // /infer and /ocr/stream. All three run on the device-agnostic
  // InferFunc/InferOneFunc seam and are registered on EVERY backend, so the
  // flag had no reader left — while its comment still told the next person
  // those endpoints were GPU-only. A dead field is survivable; a dead field
  // that documents the opposite of what the server does is not.
  // gRPC response encoding: "json_bytes" (default) or "structured".
  std::string grpc_response_mode = "json_bytes";

  // ---- pdf ----
  // honored_auto_verified: the GPU build runs auto_verified as its own path;
  // the CPU build aliases it to auto. auto_verified is listed in the modes
  // array only when it is honored as a distinct path.
  bool honored_auto_verified = false;
  int  pdf_default_dpi = 100;         // render DPI when ?dpi= is absent
  int  max_pdf_pages = 2000;          // cfg.max_pdf_pages

  // ---- limits ----
  int max_body_mb = 100;              // cfg.max_body_mb
  int max_image_dim = 16384;          // cfg.max_image_dim
  int max_batch_images = 1024;        // cfg.max_batch_images
};

/// Register GET /capabilities. Returns a stable JSON document; see
/// CapabilitiesInfo for the field semantics.
// `prebuilt_json`: the document build_capabilities_json already produced for
// the gRPC HealthResponse. Passing it (rather than letting this function
// re-derive it from `info`) is what makes the "the SAME document is served by
// both transports" claim structurally true instead of true-by-coincidence —
// two independent builds from two routing-config reads can drift.
void register_capabilities_route(const CapabilitiesInfo &info,
                                 std::string prebuilt_json = {});

/// Fill a CapabilitiesInfo from the server config + the stage availability that
/// actually loaded. ONE definition, shared with the gRPC Health RPC through
/// build_capabilities_json below, so the two transports cannot advertise
/// different contracts.
[[nodiscard]] CapabilitiesInfo make_capabilities_info(
    const server::ServerConfig &cfg, bool is_gpu,
    const capability::CapabilityMask &loaded, bool profile_endpoint,
    bool honored_auto_verified, int pdf_default_dpi);

/// The capability document itself — the exact JSON GET /capabilities serves.
/// Exposed so the gRPC HealthResponse can carry the identical bytes.
[[nodiscard]] std::string build_capabilities_json(const CapabilitiesInfo &info);

/// Register the health endpoints: /health, /health/live and /health/ready
/// (the /ocr and /ocr/raw registrations live in their own route TUs).
///
/// Single caller: src/service/server/unified/server_main.cpp. There is no
/// vendor override of /ocr/raw — on-device decode (nvJPEG, vImage) is reached
/// through the `encoded_infer` seam passed in below, not by a second route
/// registrar shadowing this one.
/// readiness_check: optional callable that returns true if the server is ready.
/// Used by /health/ready to verify GPU/pipeline is responsive.
/// `pool`: when non-null, the readiness check is offloaded to the WorkPool
/// so it never blocks a Drogon event-loop thread — a blocked probe stalls
/// that IO thread and, under load, can trip the k8s readiness timeout and
/// evict a healthy-but-busy pod. Both the GPU check (a real inference on a
/// cache miss) and the CPU check (a blocking pool->acquire()) are heavy
/// enough to require this, so callers should always pass their pool.
void register_health_route(std::function<bool()> readiness_check = nullptr,
                           server::WorkPool *pool = nullptr);

// `table_available`/`formula_available` MUST reflect what this build's pipeline
// actually loaded (GPU: routing-derived; CPU: env-derived) so the fail-loud 400
// matches reality — passing the routing-name set here would silently diverge on
// the CPU build. See check_structure_backends.
/// `encoded_infer` (optional): when set, the route hands the pipeline the
/// ENCODED bytes instead of decoding first, so a backend with an on-device
/// decoder never pays a host decode plus a full-frame upload. Safe to pass
/// unconditionally — backends without one decode on the host inside the
/// pipeline, byte-identically. The post-decode bomb guard moves with it (it
/// now lives behind UnifiedOcrPipeline::run_encoded).
void register_ocr_base64_route(server::WorkPool &pool,
                                const server::InferFunc &infer,
                                const server::ImageDecoder &decode,
                                const capability::CapabilityMask &loaded,
                                const server::EncodedInferFunc &encoded_infer = {});

/// Register POST /ocr/markdown — faithful Markdown export of one page image.
/// Requires the layout model; rejects with LAYOUT_DISABLED without it. Runs the
/// same faithful-export defaults as /ocr/pdf?markdown=1 (table/formula stages
/// only when their backends are loaded).
void register_ocr_markdown_route(server::WorkPool &pool,
                                 const server::InferFunc &infer,
                                 const server::ImageDecoder &decode,
                                 const capability::CapabilityMask &loaded);

/// Register POST /infer (Tier-B) — one crop through one named or inline
/// table/formula backend. No-op when `infer_one` is empty (no pipeline pool).
void register_infer_route(server::WorkPool &pool,
                          const server::InferOneFunc &infer_one,
                          const server::ImageDecoder &decode);

void register_ocr_raw_route(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             const capability::CapabilityMask &loaded,
                             const server::EncodedInferFunc &encoded_infer = {});

/// Convenience: register /health + /ocr + /ocr/raw (CPU paths).
/// `readiness_check` is forwarded to /health/ready and should also be
/// passed to start_grpc_server so HTTP and gRPC probes agree.
void register_common_routes(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             const capability::CapabilityMask &loaded,
                             std::function<bool()> readiness_check = nullptr,
                             const server::EncodedInferFunc &encoded_infer = {});

} // namespace turbo_ocr::routes
