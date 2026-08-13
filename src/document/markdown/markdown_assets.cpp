// OpenCV-backed asset-crop helpers, kept out of the pure string renderer
// so consumers of render_markdown alone do not drag in OpenCV imgcodecs.
#include "turbo_ocr/document/markdown_export.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/base/geometry/box.h"

namespace turbo_ocr::markdown {
namespace {

// ── base64 (data-URI embed path only) ────────────────────────────────────
[[nodiscard]] std::string base64_encode(const unsigned char *p, size_t n) {
  static const char *T =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((n + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 3 <= n; i += 3) {
    uint32_t v = (p[i] << 16) | (p[i + 1] << 8) | p[i + 2];
    out += T[(v >> 18) & 63];
    out += T[(v >> 12) & 63];
    out += T[(v >> 6) & 63];
    out += T[v & 63];
  }
  if (i < n) {
    uint32_t v = p[i] << 16;
    if (i + 1 < n) v |= p[i + 1] << 8;
    out += T[(v >> 18) & 63];
    out += T[(v >> 12) & 63];
    out += (i + 1 < n) ? T[(v >> 6) & 63] : '=';
    out += '=';
  }
  return out;
}

cv::Mat crop_region(const cv::Mat &page, const turbo_ocr::Box &box) {
  if (page.empty()) return {};
  auto r = turbo_ocr::clamped_crop_rect(box, page.cols, page.rows);
  return page(cv::Rect(r[0], r[1], r[2], r[3])).clone();
}

} // namespace

int write_asset_crops(const cv::Mat &page,
                      const std::vector<MarkdownAsset> &assets,
                      const std::string &base_dir) {
  int n = 0;
  for (const auto &a : assets) {
    cv::Mat crop = crop_region(page, a.box);
    if (crop.empty()) continue;
    std::filesystem::path out = std::filesystem::path(base_dir) / a.rel_path;
    std::error_code ec;
    std::filesystem::create_directories(out.parent_path(), ec);
    if (cv::imwrite(out.string(), crop)) ++n;
  }
  return n;
}

std::string crop_to_png_data_uri(const cv::Mat &page,
                                 const turbo_ocr::Box &box) {
  cv::Mat crop = crop_region(page, box);
  if (crop.empty()) return {};
  std::vector<unsigned char> buf;
  if (!cv::imencode(".png", crop, buf) || buf.empty()) return {};
  return "data:image/png;base64," + base64_encode(buf.data(), buf.size());
}

std::string render_markdown_with_assets(const pipeline::OcrPipelineResult &res,
                                        const cv::Mat &page,
                                        const std::string &base_dir,
                                        bool embed_images,
                                        const MarkdownOptions &opts) {
  if (embed_images) {
    ImageSrcResolver resolver = [&](const MarkdownAsset &a) {
      std::string uri = crop_to_png_data_uri(page, a.box);
      return uri.empty() ? a.rel_path : uri;
    };
    return render_markdown(res, opts, nullptr, resolver);
  }
  std::vector<MarkdownAsset> assets;
  std::string md = render_markdown(res, opts, &assets);
  write_asset_crops(page, assets, base_dir);
  return md;
}

} // namespace turbo_ocr::markdown
