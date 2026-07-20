// gRPC server construction + lifecycle; declarations in
// turbo_ocr/grpc/grpc_service.h.
#include "turbo_ocr/grpc/grpc_service.h"

namespace turbo_ocr::server {

namespace detail {

GrpcHandle launch_grpc_server(std::shared_ptr<OCRServiceImpl> service,
                                      int port, const ServerConfig &cfg) {
  // MAX_BODY_MB and GRPC_CQS now sourced from ServerConfig — the HTTP path
  // pulls from the same cfg so gRPC and HTTP body caps cannot drift.
  const int max_body_mb = cfg.max_body_mb;
  // Compute in int64 so MAX_BODY_MB=2048 (= 2^31 bytes) doesn't wrap
  // signed int. gRPC's SetMax{Receive,Send}MessageSize takes int, so
  // clamp to INT_MAX (~2 GiB) — operators wanting more must split
  // requests at the application layer.
  const int64_t max_msg64 = static_cast<int64_t>(max_body_mb) * 1024 * 1024;
  const int max_msg = static_cast<int>(
      std::min<int64_t>(max_msg64, std::numeric_limits<int>::max()));
  const int cqs = cfg.grpc_cqs;

  auto address = std::format("{}:{}", cfg.host, port);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  builder.SetMaxReceiveMessageSize(max_msg);
  builder.SetMaxSendMessageSize(max_msg);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::NUM_CQS, cqs);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, cqs);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, cqs * 2);
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 1);
  builder.AddChannelArgument(GRPC_ARG_MINIMAL_STACK, 1);

  auto server = builder.BuildAndStart();
  // BuildAndStart returns null when the listener fails (port in use,
  // permission denied). Throw so main()'s startup catch turns it into a
  // logged clean exit — otherwise the Wait() thread dereferences null and
  // the process dies by segfault with a misleading "listening" log line.
  if (!server)
    throw std::runtime_error(std::format(
        "gRPC server failed to bind {} (port in use or permission denied)",
        address));
  std::cout << std::format("gRPC server listening on {} (max_body_mb={})\n",
                            address, max_body_mb);

  auto thread = std::jthread([srv = server.get(), svc = std::move(service)]() {
    srv->Wait();
  });

  return {std::move(server), std::move(thread)};
}

} // namespace detail

#ifndef USE_CPU_ONLY
/// Start gRPC server using a PipelineDispatcher (GPU path).
/// `readiness_check` is invoked from Health() so gRPC probes match
/// HTTP /health/ready behaviour. Pass {} to keep Health unconditionally OK.
GrpcHandle start_grpc_server(pipeline::PipelineDispatcher &dispatcher,
                                     const ServerConfig &cfg,
                                     render::PdfRenderer *pdf_renderer,
                                     bool layout_available,
                                     std::function<bool()> readiness_check,
                                     bool table_available,
                                     bool formula_available,
                                     std::string capabilities_json) {
  auto service = std::make_shared<OCRServiceImpl>(
      dispatcher, cfg, pdf_renderer, layout_available);
  service->set_readiness_check(std::move(readiness_check));
  service->set_structure_availability(table_available, formula_available);
  service->set_capabilities_json(std::move(capabilities_json));
  {
    const auto rtbl = backend_routing::load_routing_config();
    service->set_routing_validation(
        backend_routing::routable_backend_names(rtbl, "table"),
        backend_routing::routable_backend_names(rtbl, "formula"));
  }
  return detail::launch_grpc_server(std::move(service), cfg.grpc_port, cfg);
}
#endif

/// Start gRPC server using an InferFunc (CPU path, also usable from GPU).
GrpcHandle start_grpc_server(InferFunc infer_fn,
                                     const ServerConfig &cfg,
                                     render::PdfRenderer *pdf_renderer,
                                     bool layout_available,
                                     std::function<bool()> readiness_check,
                                     bool table_available,
                                     bool formula_available,
                                     std::string capabilities_json) {
  auto service = std::make_shared<OCRServiceImpl>(
      std::move(infer_fn), cfg, pdf_renderer, layout_available);
  service->set_readiness_check(std::move(readiness_check));
  service->set_structure_availability(table_available, formula_available);
  service->set_capabilities_json(std::move(capabilities_json));
  // CPU build: overrides are rejected before validation, but thread the sets
  // anyway so the behavior is uniform if a CPU routing path ever lands.
  {
    const auto rtbl = backend_routing::load_routing_config();
    service->set_routing_validation(
        backend_routing::routable_backend_names(rtbl, "table"),
        backend_routing::routable_backend_names(rtbl, "formula"));
  }
  return detail::launch_grpc_server(std::move(service), cfg.grpc_port, cfg);
}


} // namespace turbo_ocr::server
