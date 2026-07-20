# gRPC service implementation TUs, shared by both server targets (compiled
# per-target: USE_CPU_ONLY selects the CPU branches inside).
set(TURBO_GRPC_SRCS
    src/grpc/grpc_helpers.cpp
    src/grpc/server_launch.cpp
    src/grpc/service_core.cpp
    src/grpc/recognize_rpc.cpp
    src/grpc/recognize_batch_rpc.cpp
    src/grpc/recognize_pdf_rpc.cpp
)
