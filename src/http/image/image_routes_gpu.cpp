#include "turbo_ocr/http/image_routes.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>
#include <optional>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/markdown/markdown_export.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/server/error_codes.h"
#include "turbo_ocr/validation/request_gate.h"
#include "turbo_ocr/validation/pixel_dims.h"

using turbo_ocr::decode::NvJpegDecoder;

#include "image_internal.h"

namespace turbo_ocr::routes {

void register_image_routes(server::WorkPool &pool,
                           pipeline::PipelineDispatcher &dispatcher,
                           const server::ImageDecoder &decode,
                           bool nvjpeg_available,
                           bool layout_available,
                           bool table_available,
                           bool formula_available,
                           int max_batch_images) {
  register_ocr_raw_route_gpu(pool, dispatcher, decode, nvjpeg_available,
                             layout_available, table_available, formula_available);
  register_ocr_batch_route_gpu(pool, dispatcher, decode, nvjpeg_available,
                               layout_available, table_available, formula_available,
                               max_batch_images);
  register_ocr_pixels_route_gpu(pool, dispatcher, layout_available,
                                table_available, formula_available);
  register_ocr_markdown_route_gpu(pool, dispatcher, decode, layout_available,
                                  table_available, formula_available);
  register_infer_route_gpu(pool, dispatcher, decode);
}

} // namespace turbo_ocr::routes
