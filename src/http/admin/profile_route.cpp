#include "turbo_ocr/http/cpu_image_routes.h"

#include <atomic>
#include <format>
#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/common/log/stage_profiler.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/validation/pixel_dims.h"
#include "turbo_ocr/validation/request_gate.h"

using turbo_ocr::base64_decode;

namespace turbo_ocr::routes {

void register_profile_route() {
  drogon::app().registerHandler(
      "/profile",
      [](const drogon::HttpRequestPtr &,
         std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(prof::dump_json_and_reset());
        callback(resp);
      },
      {drogon::Get});
}

} // namespace turbo_ocr::routes
