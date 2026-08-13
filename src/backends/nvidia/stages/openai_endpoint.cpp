// OpenAIEndpoint — the CUDA-typed face of the OpenAI-compatible VLM endpoint.
//
// Everything device-free lives in turbo_ocr/analysis/vlm/openai_policy.h and is shared
// verbatim with BackendOpenAIEndpoint (src/pipeline/unified/vlm_factory.cpp), which wears
// the de-CUDA'd backend:: seam. See that header for why the split is here and
// not somewhere else. What remains below is exactly two things: the interface
// plumbing the OLD formula::/table:: seams demand, and d2h_page() — the single
// device-specific step, bringing a GpuImage into host RAM.

#include "nvidia/stages/openai_endpoint.h"

#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include "turbo_ocr/analysis/vlm/openai_policy.h"

namespace turbo_ocr::vlm {
namespace {

// The ONE thing this class does that the shared policy cannot: read the page
// out of CUDA device memory. `scratch` owns the bytes and must outlive the
// policy call that consumes the returned HostPage — the crops are PNG-encoded
// (and COPIED into the pool) before it returns, so the caller's frame is enough.
//
// Returning an empty HostPage is the decline signal: the policy then answers
// with empty/not-ok results instead of touching the pointer.
openai_policy::HostPage d2h_page(const GpuImage &page, cudaStream_t stream,
                                 std::vector<std::uint8_t> &scratch) {
  if (page.empty()) return {};
  const std::size_t need = static_cast<std::size_t>(page.rows) * page.step;
  scratch.resize(need);
  if (cudaSuccess != cudaMemcpyAsync(scratch.data(), page.data, need,
                                     cudaMemcpyDeviceToHost, stream))
    return {};
  // A sync failure leaves `scratch` partially/garbage-filled; PNG-encoding and
  // shipping crops built from undefined memory is worse than failing loud.
  if (cudaSuccess != cudaStreamSynchronize(stream)) return {};
  return openai_policy::HostPage{scratch.data(), page.cols, page.rows,
                                 page.step};
}

} // namespace

OpenAIEndpoint::OpenAIEndpoint(backend_routing::BackendSpec spec)
    : spec_(std::move(spec)) {}

bool OpenAIEndpoint::health_check() {
  return openai_policy::health_check(spec_, ready_);
}

std::string OpenAIEndpoint::parse_one(const std::string &raw) const {
  return openai_policy::parse_with(spec_.parser, raw);
}

std::function<std::string(const std::string &)>
OpenAIEndpoint::async_result_parser() const {
  // Capture the parser enum by value — the returned callable holds no pointer
  // back into this endpoint, so it stays valid after a pipeline recycle.
  const backend_routing::Parser parser = spec_.parser;
  return [parser](const std::string &raw) {
    return openai_policy::parse_with(parser, raw);
  };
}

std::vector<std::future<std::string>>
OpenAIEndpoint::submit_async(const GpuImage &page, const std::vector<Box> &boxes,
                             cudaStream_t stream) {
  // D2H once, PNG-encode each crop, submit to the global pool, and return one
  // raw-response future per box (NO await, NO parse) — the async primitive.
  std::vector<std::uint8_t> scratch;
  return openai_policy::submit_crops_async(
      spec_, ready_, boxes,
      [&] { return d2h_page(page, stream, scratch); });
}

std::vector<formula::FormulaEngineResult>
OpenAIEndpoint::run(const GpuImage &page, const std::vector<Box> &boxes,
                    cudaStream_t stream) {
  std::vector<std::uint8_t> scratch;
  auto crops = openai_policy::infer_crops(
      spec_, ready_, boxes,
      [&] { return d2h_page(page, stream, scratch); });
  return openai_policy::to_formula_results<formula::FormulaEngineResult>(
      std::move(crops));
}

std::vector<router::TableResult>
OpenAIEndpoint::run(const GpuImage &page, const std::vector<Box> &regions,
                    const std::vector<OCRResultItem> & /*page_ocr*/,
                    cudaStream_t stream) {
  std::vector<std::uint8_t> scratch;
  auto crops = openai_policy::infer_crops(
      spec_, ready_, regions,
      [&] { return d2h_page(page, stream, scratch); });
  return openai_policy::to_table_results(std::move(crops), regions);
}

} // namespace turbo_ocr::vlm
