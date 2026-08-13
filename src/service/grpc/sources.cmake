# gRPC service implementation TUs, shared by both server targets (compiled
# per-target: USE_CPU_ONLY selects the CPU branches inside).
set(TURBO_GRPC_SRCS
    src/service/grpc/grpc_helpers.cpp
    src/service/grpc/server_launch.cpp
    src/service/grpc/service_core.cpp
    src/service/grpc/recognize_rpc.cpp
    src/service/grpc/recognize_batch_rpc.cpp
    src/service/grpc/recognize_pdf_rpc.cpp
    # Transport parity: the RPCs for /ocr/markdown, /infer and /ocr/stream.
    src/service/grpc/recognize_markdown_rpc.cpp
    src/service/grpc/recognize_stream_rpc.cpp
)
