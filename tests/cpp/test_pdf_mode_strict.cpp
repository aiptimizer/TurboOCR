#include <catch_amalgamated.hpp>

#include "turbo_ocr/pdf/pdf_extraction_mode.h"

using turbo_ocr::pdf::is_valid_pdf_mode;
using turbo_ocr::pdf::parse_pdf_mode;
using turbo_ocr::pdf::PdfMode;

TEST_CASE("parse_pdf_mode maps every documented value", "[pdf_mode]") {
  CHECK(parse_pdf_mode("ocr") == PdfMode::Ocr);
  CHECK(parse_pdf_mode("geometric") == PdfMode::Geometric);
  CHECK(parse_pdf_mode("auto") == PdfMode::Auto);
  CHECK(parse_pdf_mode("auto_verified") == PdfMode::AutoVerified);
}

TEST_CASE("parse_pdf_mode falls back only for env-style lenient callers",
          "[pdf_mode]") {
  CHECK(parse_pdf_mode("", PdfMode::Auto) == PdfMode::Auto);
  CHECK(parse_pdf_mode("garbage", PdfMode::Geometric) == PdfMode::Geometric);
}

TEST_CASE("is_valid_pdf_mode is the strict request-parsing gate", "[pdf_mode]") {
  CHECK(is_valid_pdf_mode("ocr"));
  CHECK(is_valid_pdf_mode("geometric"));
  CHECK(is_valid_pdf_mode("auto"));
  CHECK(is_valid_pdf_mode("auto_verified"));
  CHECK_FALSE(is_valid_pdf_mode(""));
  CHECK_FALSE(is_valid_pdf_mode("OCR"));      // case-sensitive by contract
  CHECK_FALSE(is_valid_pdf_mode("automatic"));
  CHECK_FALSE(is_valid_pdf_mode("auto "));
}
