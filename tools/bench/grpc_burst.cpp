#include <atomic>
#include <chrono>
#include <fstream>
#include <format>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "ocr.grpc.pb.h"

// Pure burst benchmark: all threads hammer the server simultaneously
// No warmup sweep, just max throughput measurement

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: grpc_burst <target> <image_path> <num_requests> [concurrency=16]\n";
    return 1;
  }
  std::string target = argv[1];
  std::string image_path = argv[2];
  int num_requests = std::atoi(argv[3]);
  int concurrency = argc > 4 ? std::atoi(argv[4]) : 16;

  std::ifstream file(image_path, std::ios::binary);
  std::string image_bytes((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

  // Pre-create channels and stubs
  std::vector<std::unique_ptr<ocr::OCRService::Stub>> stubs(concurrency);
  for (int i = 0; i < concurrency; ++i) {
    auto ch = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    stubs[i] = ocr::OCRService::NewStub(ch);
  }

  // Warmup (5 requests)
  for (int i = 0; i < 5; ++i) {
    ocr::OCRRequest req; req.set_image(image_bytes);
    ocr::OCRResponse resp; grpc::ClientContext ctx;
    stubs[0]->Recognize(&ctx, req, &resp);
  }

  std::atomic<int> next{0};
  std::atomic<int> done{0};

  auto t0 = std::chrono::steady_clock::now();

  std::vector<std::jthread> threads;
  for (int t = 0; t < concurrency; ++t) {
    threads.emplace_back([&, t]() {
      ocr::OCRRequest req; req.set_image(image_bytes);
      while (true) {
        int idx = next.fetch_add(1);
        if (idx >= num_requests) break;
        ocr::OCRResponse resp; grpc::ClientContext ctx;
        stubs[t]->Recognize(&ctx, req, &resp);
        done.fetch_add(1);
      }
    });
  }
  threads.clear();

  auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << std::format("{:.0f}", num_requests / elapsed) << std::flush;
  return 0;
}
