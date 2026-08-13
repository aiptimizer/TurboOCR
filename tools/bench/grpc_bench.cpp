#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <format>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "ocr.grpc.pb.h"

// High-performance C++ gRPC benchmark client
// Uses thread pool with dedicated channels for maximum throughput

struct BenchResult {
  int total_requests;
  int total_detections;
  double elapsed_sec;
  double rps() const { return total_requests / elapsed_sec; }
};

BenchResult run_bench(const std::string &target, const std::string &image_bytes,
                      int num_requests, int concurrency) {
  // One channel per thread for max parallelism (no channel contention)
  std::vector<std::shared_ptr<grpc::Channel>> channels(concurrency);
  grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MINIMAL_STACK, 1);
  for (int i = 0; i < concurrency; ++i) {
    channels[i] = grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), args);
  }

  std::atomic<int> next_idx{0};
  std::atomic<int> total_det{0};
  std::atomic<int> errors{0};

  auto t0 = std::chrono::steady_clock::now();

  std::vector<std::jthread> threads;
  threads.reserve(concurrency);
  for (int t = 0; t < concurrency; ++t) {
    threads.emplace_back([&, t]() {
      auto stub = ocr::OCRService::NewStub(channels[t]);
      ocr::OCRRequest req;
      req.set_image(image_bytes);

      while (true) {
        int idx = next_idx.fetch_add(1);
        if (idx >= num_requests) break;

        ocr::OCRResponse resp;
        grpc::ClientContext ctx;
        auto status = stub->Recognize(&ctx, req, &resp);
        if (status.ok()) {
          // Handle both modes: structured (results_size) and json_bytes (num_detections)
          int det = resp.results_size() > 0 ? resp.results_size() : resp.num_detections();
          total_det.fetch_add(det);
        } else {
          if (errors.fetch_add(1) == 0) {
            std::cerr << std::format("  First error: code={} msg={}\n",
                                     (int)status.error_code(), status.error_message());
          }
        }
      }
    });
  }
  threads.clear(); // join all

  auto t1 = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();

  if (errors > 0) {
    std::cerr << std::format("  {} errors out of {} requests", errors.load(), num_requests) << '\n';
  }

  return {num_requests, total_det.load(), elapsed};
}

int main(int argc, char **argv) {
  std::string target = "localhost:50051";
  std::string image_path = "test_images/cord_000.png";

  if (argc > 1) target = argv[1];
  if (argc > 2) image_path = argv[2];

  // Load image
  std::ifstream file(image_path, std::ios::binary);
  if (!file) {
    std::cerr << "Cannot open: " << image_path << '\n';
    return 1;
  }
  std::string image_bytes((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
  std::cout << std::format("Image: {} ({} bytes)", image_path, image_bytes.size()) << '\n';

  // Warmup
  std::cout << "Warming up..." << '\n';
  run_bench(target, image_bytes, 50, 4);

  // Sweep concurrency
  std::cout << std::format("\n=== C++ gRPC benchmark → {} ===", target) << '\n';
  for (int c : {1, 2, 4, 8, 12, 16, 24, 32, 48, 64}) {
    int n = std::max(200, c * 50);
    auto r = run_bench(target, image_bytes, n, c);
    std::cout << std::format("  c={:>3}: {:>7.0f} img/s | avg {:>6.1f}ms | det={:.1f}",
                             c, r.rps(), r.elapsed_sec / r.total_requests * 1000,
                             (double)r.total_detections / r.total_requests) << '\n';
  }

  return 0;
}
