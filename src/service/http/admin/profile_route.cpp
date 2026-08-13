#include "turbo_ocr/service/http/image_routes.h"

#include <atomic>
#include <format>
#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/base/log/stage_profiler.h"
#include "turbo_ocr/image/image_config.h"
#include "turbo_ocr/image/image_dims.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/service/validation/pixel_dims.h"
#include "turbo_ocr/service/validation/request_gate.h"

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
