// MetalImage implementation (see metal_image.h).

#import "apple/memory/metal_image.h"
#import "apple/support/metal_common.h"
#import "apple/support/apple_contention.h"

#import <Foundation/Foundation.h>

#include <opencv2/core.hpp>

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "turbo_ocr/base/env_utils.h"

namespace turbo_ocr::apple {

// --- texture registry -------------------------------------------------------

namespace {
struct TexEntry {
  id<MTLTexture> tex = nil;
  bool dirty = true; // pixels not yet packed for the CURRENT page content
  // PAGE-IDENTITY AUDIT (TURBO_APPLE_PAGE_AUDIT=1). Fingerprint of the source
  // BGR8 bytes at the moment this texture was last packed, plus the geometry
  // it was packed for. On a cache HIT the audit re-fingerprints the source: a
  // mismatch means the texture we are about to sample holds SOME OTHER PAGE's
  // pixels, which is the wrong-page hypothesis observed directly rather than
  // inferred from a transcript. Off by default (a fingerprint per page is
  // cheap but not free).
  std::uint64_t fp = 0;
  int fp_rows = 0, fp_cols = 0;
};
struct TexReg {
  std::mutex m;
  std::map<const void *, TexEntry> by_data;
};
TexReg &texreg() {
  static TexReg r;
  return r;
}

bool page_audit_enabled() {
  static const bool on = [] {
    const std::string e = env::env_or("TURBO_APPLE_PAGE_AUDIT", "");
    return !e.empty() && e != "0";
  }();
  return on;
}

// Strided FNV-1a over the page bytes. Not a hash of every byte: at ~2 MB a page
// that would be measurable, and any wrong-page substitution changes thousands of
// widely separated bytes, so a few hundred probes detect it with certainty.
std::uint64_t page_fingerprint(const backend::ImageView &img) {
  const auto *p = static_cast<const std::uint8_t *>(img.data);
  if (!p) return 0;
  const std::size_t bytes = img.step * (std::size_t)img.rows;
  std::uint64_t h = 1469598103934665603ULL;
  const std::size_t stride = std::max<std::size_t>(1, bytes / 512);
  for (std::size_t i = 0; i < bytes; i += stride) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  h ^= bytes;
  h *= 1099511628211ULL;
  return h ? h : 1;
}

std::atomic<unsigned long long> g_audit_hits{0}, g_audit_stale{0};
} // namespace

void page_audit_report(const char *where) {
  if (!page_audit_enabled()) return;
  NSLog(@"[apple] PAGE AUDIT (%s): texture cache hits=%llu STALE(wrong page)=%llu "
        @"live texture entries=%zu",
        where ? where : "", g_audit_hits.load(), g_audit_stale.load(),
        registered_texture_count());
}

void register_texture(const void *view_data, id<MTLTexture> tex) {
  auto &r = texreg();
  TURBO_APPLE_LOCK(tex_register_lock, r.m);
  r.by_data[view_data] = TexEntry{tex, /*dirty*/ false};
}

void invalidate_texture(const void *view_data) {
  auto &r = texreg();
  TURBO_APPLE_LOCK(tex_invalidate_lock, r.m);
  auto it = r.by_data.find(view_data);
  if (it != r.by_data.end()) it->second.dirty = true;
}
void unregister_texture(const void *view_data) {
  auto &r = texreg();
  TURBO_APPLE_LOCK(tex_unregister_lock, r.m);
  r.by_data.erase(view_data);
}
id<MTLTexture> texture_for(const void *view_data) {
  auto &r = texreg();
  TURBO_APPLE_LOCK(tex_lookup_lock, r.m);
  auto it = r.by_data.find(view_data);
  return it == r.by_data.end() ? nil : it->second.tex;
}
std::size_t registered_texture_count() {
  auto &r = texreg();
  std::lock_guard<std::mutex> lk(r.m);
  return r.by_data.size();
}

// --- MetalImage -------------------------------------------------------------

MetalImage::~MetalImage() {
  if (buf_) unregister_texture(buf_.contents);
  tex_ = nil;
  buf_ = nil;
}

MetalImage::MetalImage(MetalImage &&o) noexcept
    : tex_(o.tex_), buf_(o.buf_), rows_(o.rows_), cols_(o.cols_) {
  o.tex_ = nil;
  o.buf_ = nil;
  o.rows_ = o.cols_ = 0;
}

MetalImage &MetalImage::operator=(MetalImage &&o) noexcept {
  if (this != &o) {
    if (buf_) unregister_texture(buf_.contents);
    tex_ = o.tex_;
    buf_ = o.buf_;
    rows_ = o.rows_;
    cols_ = o.cols_;
    o.tex_ = nil;
    o.buf_ = nil;
    o.rows_ = o.cols_ = 0;
  }
  return *this;
}

MetalImage MetalImage::from_host_bgr(const cv::Mat &bgr) {
  MetalImage img;
  if (bgr.empty() || bgr.type() != CV_8UC3) return img;
  const int h = bgr.rows, w = bgr.cols;
  img.rows_ = h;
  img.cols_ = w;

  @autoreleasepool {
    // Canonical BGR8 Shared buffer (tight rows: step = w*3).
    const size_t step = (size_t)w * 3;
    img.buf_ = [mtl_device() newBufferWithLength:step * h
                                         options:MTLResourceStorageModeShared];
    auto *dst = static_cast<std::uint8_t *>(img.buf_.contents);
    for (int y = 0; y < h; ++y)
      std::memcpy(dst + (size_t)y * step, bgr.ptr(y), step);

    // RGBA8 texture, B/R swapped (texture .rgb == R,G,B) — identical to the
    // proven upload in tools/probes/apple/mps_ocr.mm:63-69.
    MTLTextureDescriptor *td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:w
                                                          height:h
                                                       mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    img.tex_ = [mtl_device() newTextureWithDescriptor:td];
    std::vector<std::uint8_t> px((size_t)w * h * 4);
    for (int y = 0; y < h; ++y) {
      const cv::Vec3b *row = bgr.ptr<cv::Vec3b>(y);
      for (int x = 0; x < w; ++x) {
        size_t i = ((size_t)y * w + x) * 4;
        px[i + 0] = row[x][2]; // R
        px[i + 1] = row[x][1]; // G
        px[i + 2] = row[x][0]; // B
        px[i + 3] = 255;
      }
    }
    [img.tex_ replaceRegion:MTLRegionMake2D(0, 0, w, h)
                mipmapLevel:0
                  withBytes:px.data()
                bytesPerRow:(size_t)w * 4];
  }
  register_texture(img.buf_.contents, img.tex_);
  return img;
}

backend::ImageView MetalImage::view() const {
  return backend::ImageView{
      .data = buf_ ? buf_.contents : nullptr,
      .step = (size_t)cols_ * 3,
      .rows = rows_,
      .cols = cols_,
      .kind = backend::DeviceKind::Metal,
  };
}

id<MTLTexture> ensure_texture(const backend::ImageView &img,
                              id<MTLCommandBuffer> cb) {
  if (img.empty() || !cb) return nil;

  // Reuse the cached texture when it is both the right SHAPE and known to hold
  // the current page's pixels.
  //
  // DIMENSION GUARD (was a silent wrong-page bug): the registry is keyed by a raw
  // device pointer, so a recycled .contents address would otherwise HIT a
  // previous page's texture. DIRTY GUARD: the upload buffer is now REUSED across
  // pages by the shared pipeline, so a same-size page hits a correctly-shaped but
  // stale texture — MetalAllocator::copy_h2d marks it dirty and we repack below.
  const bool audit = page_audit_enabled();
  const std::uint64_t fp = audit ? page_fingerprint(img) : 0;

  id<MTLTexture> tex = nil;
  {
    auto &r = texreg();
    TURBO_APPLE_LOCK(tex_lookup_lock2, r.m);
    auto it = r.by_data.find(img.data);
    if (it != r.by_data.end()) {
      id<MTLTexture> t = it->second.tex;
      if ((int)t.width == img.cols && (int)t.height == img.rows) {
        if (!it->second.dirty) {
          if (audit) {
            g_audit_hits.fetch_add(1, std::memory_order_relaxed);
            if (it->second.fp != fp || it->second.fp_rows != img.rows ||
                it->second.fp_cols != img.cols) {
              g_audit_stale.fetch_add(1, std::memory_order_relaxed);
              NSLog(@"[apple] PAGE AUDIT: STALE TEXTURE for %p — cached pack was "
                    @"fp=%llx %dx%d, source now fp=%llx %dx%d. The kernel would "
                    @"have sampled ANOTHER PAGE.",
                    img.data, (unsigned long long)it->second.fp,
                    it->second.fp_cols, it->second.fp_rows,
                    (unsigned long long)fp, img.cols, img.rows);
            }
          }
          return t;
        }
        tex = t;                    // right shape, stale pixels -> repack in place
        it->second.dirty = false;
      } else {
        r.by_data.erase(it);        // wrong shape -> rebuild
      }
    }
  }

  std::size_t off = 0;
  id<MTLBuffer> buf = resolve_buffer(img.data, &off);
  if (!buf) return nil;

  TURBO_APPLE_STAT(tex_pack_total);
  @autoreleasepool {
    if (!tex) {
      TURBO_APPLE_STAT(tex_new_texture);
      MTLTextureDescriptor *td =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                             width:img.cols
                                                            height:img.rows
                                                         mipmapped:NO];
      td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      tex = [mtl_device() newTextureWithDescriptor:td];
      register_texture(img.data, tex); // registers CLEAN; we pack it below
    }

    // Encode the pack onto the CALLER's command buffer. Compute encoders on one
    // command buffer run in submission order, so the kernel the caller encodes
    // next is guaranteed to sample packed pixels — no commit, no host wait, and
    // no process-global queue.
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:mtl_pipeline("pack_bgr8_to_rgba")];
    [enc setBuffer:buf offset:off atIndex:0];
    std::uint32_t dims[3] = {(std::uint32_t)img.cols, (std::uint32_t)img.rows,
                             (std::uint32_t)img.step};
    [enc setBytes:dims length:sizeof(dims) atIndex:1];
    [enc setTexture:tex atIndex:0];
    [enc dispatchThreads:MTLSizeMake(img.cols, img.rows, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
    if (audit) { // record WHICH page's bytes this texture was packed from
      auto &r = texreg();
      std::lock_guard<std::mutex> lk(r.m);
      auto it = r.by_data.find(img.data);
      if (it != r.by_data.end()) {
        it->second.fp = fp;
        it->second.fp_rows = img.rows;
        it->second.fp_cols = img.cols;
      }
    }
    return tex;
  }
}

} // namespace turbo_ocr::apple
