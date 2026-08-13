#include <catch_amalgamated.hpp>

#include <map>
#include <set>
#include <string>

#include "turbo_ocr/service/validation/request_validation.h"

using namespace turbo_ocr::server;
using turbo_ocr::backend_routing::RequestRouting;

namespace {

using Params = std::map<std::string, std::string>;

const std::set<std::string> kTables{"table-env"};
const std::set<std::string> kFormulas{"formula-env"};

ValidationError run(const Params &params, const EndpointSpec &spec,
                    bool strict = false, RequestRouting *out = nullptr) {
  RequestRouting scratch;
  return validate_params(params, spec, kTables, kFormulas, strict,
                         out ? out : &scratch);
}

EndpointSpec image_spec() {
  EndpointSpec s;
  s.routing = RoutingSupport::kSupported;
  return s;
}

} // namespace

TEST_CASE("supported params pass in both modes", "[request_validation]") {
  const Params p{{"layout", "1"}, {"tables", "1"}, {"text", "1"}};
  CHECK(run(p, image_spec(), /*strict=*/false).ok());
  CHECK(run(p, image_spec(), /*strict=*/true).ok());
}

TEST_CASE("known-but-unsupported param: lenient ignores, strict rejects",
          "[request_validation]") {
  // dpi is a real API param (PDF endpoints) but a harmless no-op on image
  // endpoints — dropping it does not falsify the response, so lenient mode
  // tolerates it (v3.4 compatibility). Strict mode is loud.
  const Params p{{"dpi", "150"}};
  CHECK(run(p, image_spec(), /*strict=*/false).ok());
  // Lenient tolerance reports what it ignored (deprecation surface).
  {
    RequestRouting scratch;
    std::vector<std::string> ignored;
    CHECK(validate_params(p, image_spec(), kTables, kFormulas,
                          /*strict=*/false, &scratch, &ignored)
              .ok());
    REQUIRE(ignored.size() == 1);
    CHECK(ignored[0] == "dpi");
  }
  auto e = run(p, image_spec(), /*strict=*/true);
  CHECK_FALSE(e.ok());
  CHECK(e.code == "INVALID_PARAMETER");
  CHECK(e.message.find("dpi") != std::string::npos);
  CHECK(e.message.find("not supported on this endpoint") != std::string::npos);
  // The same param on a PDF spec is category 1 (supported).
  EndpointSpec pdf;
  pdf.pdf_options = true;
  CHECK(run(p, pdf, /*strict=*/true).ok());
}

TEST_CASE("unknown param: lenient ignores, strict rejects",
          "[request_validation]") {
  const Params p{{"bogus_knob", "1"}};
  CHECK(run(p, image_spec(), /*strict=*/false).ok());
  auto e = run(p, image_spec(), /*strict=*/true);
  CHECK_FALSE(e.ok());
  CHECK(e.message.find("bogus_knob") != std::string::npos);
}

TEST_CASE("routing supported: valid name populates, unknown name rejects",
          "[request_validation]") {
  RequestRouting out;
  CHECK(run({{"route_table", "table-env"}}, image_spec(), false, &out).ok());
  CHECK(out.table == "table-env");

  auto e = run({{"route_table", "nope"}}, image_spec());
  CHECK_FALSE(e.ok());
  CHECK(e.code == "ROUTING_UNKNOWN_OVERRIDE");

  auto e2 = run({{"route_formula", "nope"}}, image_spec());
  CHECK(e2.code == "ROUTING_UNKNOWN_OVERRIDE");
}

TEST_CASE("routing unsupported: any override rejects with the spec's reason",
          "[request_validation]") {
  EndpointSpec pdf;
  pdf.pdf_options = true;
  pdf.routing_unsupported_reason = kRoutingUnsupportedPdf;
  // Even a name that WOULD be valid is rejected: the endpoint cannot honor
  // it, and validating-then-ignoring is a silent failure.
  auto e = run({{"route_table", "table-env"}}, pdf);
  CHECK_FALSE(e.ok());
  CHECK(e.code == "INVALID_PARAMETER");
  CHECK(e.message == kRoutingUnsupportedPdf);
  // No override at all is always fine.
  CHECK(run({{"layout", "1"}}, pdf).ok());
}

TEST_CASE("route params never hit the generic unsupported branch",
          "[request_validation]") {
  // A spec without routing support must reject route params with the policy
  // reason, not the generic "not supported on this endpoint" text — callers
  // need to know WHY (build limitation vs endpoint limitation).
  EndpointSpec cpu;
  cpu.routing = RoutingSupport::kUnsupported;
  cpu.routing_unsupported_reason = kRoutingUnsupportedCpu;
  auto e = run({{"route_formula", "x"}}, cpu);
  CHECK(e.message == kRoutingUnsupportedCpu);
}

TEST_CASE("spec flags derive the allowed set", "[request_validation]") {
  EndpointSpec pixels = image_spec();
  pixels.pixel_dims = true;
  CHECK(run({{"width", "100"}, {"height", "50"}, {"channels", "3"}}, pixels,
            /*strict=*/true)
            .ok());
  // The same params on a non-pixels endpoint are known-but-unsupported:
  // harmless no-ops, so lenient tolerates and strict rejects.
  CHECK(run({{"width", "100"}}, image_spec(), /*strict=*/false).ok());
  CHECK_FALSE(run({{"width", "100"}}, image_spec(), /*strict=*/true).ok());

  EndpointSpec md;
  md.ocr_options = false;
  md.markdown_embed = true;
  CHECK(run({{"embed", "1"}}, md, /*strict=*/true).ok());
  // OCR options on a fixed-purpose endpoint follow the same lenient/strict
  // split (dropping them never falsifies the fixed-purpose response).
  CHECK(run({{"layout", "1"}}, md, /*strict=*/false).ok());
  CHECK(run({{"layout", "1"}}, md, /*strict=*/true).message.find(
            "not supported on this endpoint") != std::string::npos);
}

TEST_CASE("is_known_param covers exactly the group tables",
          "[request_validation]") {
  for (const char *p : {"layout", "reading_order", "as_blocks", "tables",
                        "formulas", "text", "route_table", "route_formula",
                        "width", "height", "channels", "dpi", "mode",
                        "markdown", "as_pages", "images", "format", "lossless",
                        "png_compression", "quality", "max_side", "autorotate",
                        "output", "min_confidence", "embed"})
    CHECK(is_known_param(p));
  CHECK_FALSE(is_known_param("not_a_param"));
}

TEST_CASE("apply_routing_override is shared by body and query paths",
          "[request_validation]") {
  // The /ocr JSON `routing{}` body field and the gRPC request fields go
  // through this same function; identical inputs must yield identical
  // verdicts to the query-param path above.
  RequestRouting out;
  CHECK(apply_routing_override("table-env", "formula-env", image_spec(),
                               kTables, kFormulas, &out)
            .ok());
  CHECK(out.table == "table-env");
  CHECK(out.formula == "formula-env");
  CHECK(apply_routing_override("", "", image_spec(), kTables, kFormulas, &out)
            .ok());
  CHECK_FALSE(apply_routing_override("nope", "", image_spec(), kTables,
                                     kFormulas, &out)
                  .ok());
}
