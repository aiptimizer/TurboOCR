#pragma once

// MetalImage — an owning, device-resident page image for the Apple backend.
//
// The Backend seam passes images as backend::ImageView (a non-owning {void* data,
// step, rows, cols, kind} descriptor). On Metal a page needs TWO representations:
//   * an RGBA8 MTLTexture, for the warp/resize kernels' hardware bilinear
//     sampler (tools/probes/apple/warp.metal), and
//   * a Shared MTLBuffer of canonical interleaved BGR8 (image_view.h layout), so
//     ImageView::data is a real, resolvable Metal pointer.
// MetalImage owns both and hands out view(). A small registry maps an
// ImageView::data pointer back to its texture, so IKernels::warp_crops /
// resize_normalize can recover the sampler source from a bare ImageView. If a
// Metal ImageView has no registered texture (e.g. a decoded buffer produced
// elsewhere), the kernels lazily pack one via the pack_bgr8_to_rgba shader.
//
// This is the ImageView(Metal) backing the plan asks for — the generalization of
// GpuImage to a Metal MTLTexture/MTLBuffer.

#include <cstddef>

#include "turbo_ocr/backend/image_view.h"

namespace cv { class Mat; }

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace turbo_ocr::apple {

class MetalImage {
public:
  MetalImage() = default;
  ~MetalImage();
  MetalImage(MetalImage &&) noexcept;
  MetalImage &operator=(MetalImage &&) noexcept;
  MetalImage(const MetalImage &) = delete;
  MetalImage &operator=(const MetalImage &) = delete;

  // Upload a host BGR image (cv::IMREAD_COLOR layout — the pipeline's canonical
  // decode format) into a resident RGBA8 texture (B/R swapped, matching
  // tools/probes/apple/mps_ocr.mm:66-69) + a BGR8 Shared buffer. `bgr` must be 8UC3.
  static MetalImage from_host_bgr(const cv::Mat &bgr);

  [[nodiscard]] backend::ImageView view() const;
  [[nodiscard]] bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }
  [[nodiscard]] int rows() const noexcept { return rows_; }
  [[nodiscard]] int cols() const noexcept { return cols_; }

#ifdef __OBJC__
  [[nodiscard]] id<MTLTexture> texture() const noexcept { return tex_; }
  [[nodiscard]] id<MTLBuffer> buffer() const noexcept { return buf_; }
#endif

private:
#ifdef __OBJC__
  id<MTLTexture> tex_ = nil;
  id<MTLBuffer> buf_ = nil;
#endif
  int rows_ = 0;
  int cols_ = 0;
};

#ifdef __OBJC__
// Texture registry: an RGBA8 sampler-source texture keyed by an ImageView::data
// pointer. MetalImage registers itself; kernels resolve via texture_for().
void register_texture(const void *view_data, id<MTLTexture> tex);
void unregister_texture(const void *view_data);
id<MTLTexture> texture_for(const void *view_data);

// Mark the sampler texture cached for `view_data` as holding STALE pixels.
//
// The texture is a GPU-side RGBA8 copy of the BGR8 page buffer, so it is only
// valid until someone writes new pixels into that buffer. When page buffers were
// allocated and freed per image the key changed every page and staleness was
// impossible; the pipeline now REUSES one upload buffer per replica, so the
// moment of invalidation has to be signalled explicitly. MetalAllocator::copy_h2d
// (the one path that writes a page into a device buffer) calls this.
void invalidate_texture(const void *view_data);
// PAGE-IDENTITY AUDIT (TURBO_APPLE_PAGE_AUDIT=1). ensure_texture fingerprints
// the source page bytes when it packs a texture and re-checks the fingerprint on
// every cache HIT; a mismatch means the kernel about to run would sample ANOTHER
// PAGE and is logged as "STALE TEXTURE". Call this to print the running totals.
void page_audit_report(const char *where);

// Live entries — a leak/aliasing probe. MUST stay bounded by the number of
// pages IN FLIGHT, never grow with the number of pages PROCESSED.
std::size_t registered_texture_count();

// Recover (or lazily build) the sampler-source texture for a Metal ImageView:
// returns the registered texture, else packs the BGR8 buffer behind `img.data`
// into a fresh RGBA8 texture via pack_bgr8_to_rgba (kept resident). Returns nil
// only if `img` is not a resolvable Metal buffer.
//
// The pack is ENCODED ON `cb` — the caller's own command buffer — so it is
// ordered with the kernel that samples the texture and costs no synchronization.
// It used to run on a PROCESS-GLOBAL MTLCommandQueue followed by
// waitUntilCompleted: one blocking host round-trip per page on a queue shared by
// every replica in the process. MEASURED at K=24: 11.9 ms of blocked time per
// page (12% of all thread time), against 1.0 ms at K=8 — i.e. it degraded with
// concurrency, which is exactly the in-process serialization signature.
id<MTLTexture> ensure_texture(const backend::ImageView &img,
                              id<MTLCommandBuffer> cb);
#endif

} // namespace turbo_ocr::apple
