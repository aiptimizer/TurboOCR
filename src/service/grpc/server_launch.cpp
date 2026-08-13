// gRPC server construction + lifecycle; declarations in
// turbo_ocr/service/grpc/grpc_service.h.
#include "turbo_ocr/service/grpc/grpc_service.h"
#include "turbo_ocr/service/validation/request_gate.h" // routing_name_sets

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


/// Start gRPC server using an InferFunc (CPU path, also usable from GPU).
GrpcHandle start_grpc_server(InferFunc infer_fn,
                                     const ServerConfig &cfg,
                                     render::PdfRenderer *pdf_renderer,
                                     const capability::CapabilityMask &loaded,
                                     std::function<bool()> readiness_check,
                                     std::string capabilities_json,
                                     OrientFunc orient_fn,
                                     EncodedInferFunc encoded_infer_fn,
                                     InferOneFunc infer_one_fn) {
  // SERVICE-LOCAL NARROWING (mirror of the HTTP /ocr/pdf route): on this
  // InferFunc path autorotate is applied through `orient_fn`, so a server that
  // loaded the doc-orientation model but launched gRPC without an OrientFunc
  // genuinely cannot honour autorotate=true here. Clearing the bit makes the
  // gate reject it with AUTOROTATE_DISABLED instead of silently not rotating.
  capability::CapabilityMask grpc_loaded = loaded;
  if (!orient_fn)
    grpc_loaded.set(capability::CapabilityId::DocOrientation, false);
  auto service = std::make_shared<OCRServiceImpl>(
      std::move(infer_fn), cfg, pdf_renderer, grpc_loaded);
  service->set_orient_fn(std::move(orient_fn));
  // Optional; when the caller has a pipeline pool it should pass
  // make_encoded_infer_func(pool) so the image RPCs can keep the bytes encoded
  // all the way to a backend that decodes on-device. Absent, they host-decode.
  service->set_encoded_infer_fn(std::move(encoded_infer_fn));
  // Powers InferOne; absent, that RPC answers UNIMPLEMENTED rather than crash.
  service->set_infer_one_fn(std::move(infer_one_fn));
  service->set_readiness_check(std::move(readiness_check));
  service->set_capabilities_json(std::move(capabilities_json));
  // CPU build: overrides are rejected before validation, but thread the sets
  // anyway so the behavior is uniform if a CPU routing path ever lands.
  {
    // THE shared derivation (request_gate.h::routing_name_sets), same as every
    // HTTP route uses. It was hand-rolled here and at three HTTP sites; four
    // copies of "which backend names may a ?route_table= override name" is four
    // chances for gRPC to accept a name HTTP rejects.
    const server::RoutingNameSets routes = server::routing_name_sets();
    service->set_routing_validation(routes.table, routes.formula);
  }
  return detail::launch_grpc_server(std::move(service), cfg.grpc_port, cfg);
}


} // namespace turbo_ocr::server
