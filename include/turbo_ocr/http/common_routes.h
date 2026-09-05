#pragma once

#include "turbo_ocr/server/bootstrap/server_config.h"
#include <string>

#include "turbo_ocr/server/bootstrap/server_config.h"
#include "turbo_ocr/server/service_fns.h"
#include "turbo_ocr/server/work_pool.h"

namespace turbo_ocr::routes {

/// Inputs for the GET /capabilities endpoint (M6 cross-build parity).
/// Each binary fills this from its already-resolved ServerConfig + the
/// runtime-availability bools it computed at startup, then hands it to
/// register_capabilities_route. Kept as an explicit POD (rather than the
/// whole ServerConfig) so the contract is exactly the advertised surface
/// and adding a field here can't silently leak an unrelated config value.
///
/// `/capabilities` is purely additive: it advertises the known GPU/CPU
/// build divergences (which the two binaries deliberately keep — see M6)
/// so a client can discover, without trial requests, which features the
/// deployment it reached actually honors. It changes no existing default,
/// response shape, or endpoint behavior.
struct CapabilitiesInfo {
  bool is_gpu = false;                // build flavor: gpu vs cpu

  // ---- features ----
  bool layout_available = false;      // layout stage loaded
  bool table_available = false;       // a table backend loaded (tables=1 works)
  bool formula_available = false;     // a formula backend loaded (formulas=1 works)
  bool autorotate_available = false;  // doc-orientation model loaded
  bool profile_endpoint = false;      // GET /profile registered (CPU only)
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

/// Register GET /capabilities. Registered by BOTH binaries so a client can
/// query the contract of whichever deployment it reached. Returns a stable
/// JSON document; see CapabilitiesInfo for the field semantics.
void register_capabilities_route(const CapabilitiesInfo &info);

/// Fill a CapabilitiesInfo from the server config + the stage availability the
/// build actually loaded. One definition for both mains, so the advertised
/// document can never drift between the builds except through these explicit
/// arguments.
[[nodiscard]] CapabilitiesInfo make_capabilities_info(
    const server::ServerConfig &cfg, bool is_gpu, bool layout_available,
    bool table_available, bool formula_available, bool autorotate_available,
    bool profile_endpoint, bool honored_auto_verified, int pdf_default_dpi);

/// The capability document itself — the exact JSON GET /capabilities serves.
/// Exposed so the gRPC HealthResponse can carry the identical bytes.
[[nodiscard]] std::string build_capabilities_json(const CapabilitiesInfo &info);

/// Register /health, /ocr (base64 JSON), /ocr/raw (CPU decode path).
/// The GPU binary overrides /ocr/raw with its own nvJPEG version in
/// image_routes, so register_common_routes is only used by cpu_main.
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
/// `jpeg_infer` (may be empty) runs JPEG bytes through the GPU-direct decode
/// + inference path on the replica; when empty, JPEG is decoded by `decode`.
void register_ocr_base64_route(server::WorkPool &pool,
                                const server::InferFunc &infer,
                                const server::JpegInferFunc &jpeg_infer,
                                const server::ImageDecoder &decode,
                                bool layout_available,
                                bool table_available,
                                bool formula_available);

void register_ocr_raw_route(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available,
                             bool table_available,
                             bool formula_available);

/// Convenience: register /health + /ocr + /ocr/raw (CPU paths).
/// `readiness_check` is forwarded to /health/ready and should also be
/// passed to start_grpc_server so HTTP and gRPC probes agree.
void register_common_routes(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available,
                             bool table_available,
                             bool formula_available,
                             std::function<bool()> readiness_check = nullptr);

} // namespace turbo_ocr::routes
