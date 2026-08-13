#include "nvidia/stages/slanext_table_recognizer.h"

#include <iostream>
#include <memory>

#include "nvidia/stages/openai_endpoint.h"
#include "turbo_ocr/backend/routing_config.h"
#include "nvidia/stages/table_recognizer.h"
#include "nvidia/stages/vlm_table_recognizer.h"

namespace turbo_ocr::table {

std::unique_ptr<ITableRecognizer> make_table_recognizer(std::string_view backend) {
  if (backend == "slanext")  return std::make_unique<SlanextTableRecognizer>();
  if (backend == "vlm")      return std::make_unique<VLMTableRecognizer>();
  std::cerr << "[TableRecognizer] unknown TABLE_BACKEND='" << backend
            << "' (expected 'slanext' or 'vlm')\n";
  return nullptr;
}

std::unique_ptr<ITableRecognizer>
make_table_recognizer(const backend_routing::BackendSpec &spec) {
  if (spec.kind == backend_routing::Kind::Openai)
    return std::make_unique<vlm::OpenAIEndpoint>(spec);
  return make_table_recognizer(std::string_view{spec.engine});
}

} // namespace turbo_ocr::table
