// MetalKernels implementation (see metal_kernels.h).

#import "apple/kernels_metal/metal_kernels.h"
#import "apple/support/metal_common.h"
#import "apple/queue/metal_device_queue.h"
#import "apple/memory/metal_image.h"

#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>

#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/core/norm_params.h" // SHARED norm factories
#include "turbo_ocr/analysis/detection/det_postprocess.h"

namespace turbo_ocr::apple {

// RNParams — must match the Metal `struct RNParams` field order/size exactly
// (shaders.metal): 4 uint + 6 float + 2 uint, tightly packed (48 bytes).
namespace {
struct RNParams {
  std::uint32_t dst_w, dst_h, src_w, src_h;
  float mean0, mean1, mean2;
  float istd0, istd1, istd2;
  std::uint32_t order;     // 0 = BGR planes, 1 = RGB planes
  std::uint32_t letterbox; // 0 = stretch, 1 = preserve-AR + pad
};
static_assert(sizeof(RNParams) == 48, "RNParams must match the Metal layout");
} // namespace

MetalKernels::MetalKernels() = default;
MetalKernels::~MetalKernels() = default;

backend::KernelCaps MetalKernels::caps() const {
  backend::KernelCaps c;
  c.device = backend::DeviceKind::Metal;
  c.decode_image = true;   // host imdecode + resident upload
  c.resize_normalize = true;
  c.warp_crops = true;
  c.threshold = true;
  c.db_postprocess = false; // HOST fallback (CCL/unclip has no Metal primitive)
  c.argmax = true;
  c.preprocess_region = false; // TODO(apple-fused-region-preproc)
  // PARAMETER CONTRACT (kernels.h): declare exactly what this backend honours.
  c.params.norm_mean_std = true;        // resize_normalize is fully parameterized
  // warp.metal BAKES rec's (v/127.5 - 1) in. That is the one distribution both
  // rec and cls want, so nothing is lost today — but the op used to accept a
  // NormParams and ignore it, which is exactly the silent substitution the
  // contract forbids. It now whitelists norm::rec_norm() and refuses the rest.
  c.params.norm_mean_std_warp = false;
  c.params.norm_channel_order = true;   // shaders.metal honours params.order
  c.params.norm_letterbox = false;      // see the letterbox note in kernels.h
  c.params.db_oriented = true;          // extract_boxes_from_bitmap = minAreaRect
  c.params.db_axis_aligned = false;     // ...and it CANNOT emit AABBs
  c.params.db_expand_limits = false;    // contour path has no expand clamp
  c.params.db_side_limits = true;
  c.params.db_max_components = false;   // contour path has no component budget
  return c;
}

backend::ImageView MetalKernels::decode_image(const std::uint8_t *data,
                                              std::size_t len,
                                              backend::DeviceQueue &) {
  // NATIVE decode via ImageIO. On Apple silicon ImageIO dispatches JPEG to the
  // hardware media engine rather than running libjpeg on the CPU, and it also
  // covers HEIC/TIFF/GIF that OpenCV may not be built for. The decoded pixels
  // still go through MetalImage::from_host_bgr so the resident RGBA8 TEXTURE the
  // warp kernel samples is built exactly as before — this replaces the decoder,
  // not the residency path.
  //
  // Falls back to cv::imdecode when ImageIO declines. The seam permits an empty
  // return ("decoder declined"), but returning empty for a format OpenCV handles
  // would turn a capability upgrade into a format regression.
  @autoreleasepool {
    NSData *nsd = [NSData dataWithBytesNoCopy:const_cast<std::uint8_t *>(data)
                                       length:len
                                 freeWhenDone:NO];
    CGImageSourceRef src =
        CGImageSourceCreateWithData((__bridge CFDataRef)nsd, nullptr);
    if (src) {
      CGImageRef cg = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
      CFRelease(src);
      if (cg) {
        const int w = (int)CGImageGetWidth(cg);
        const int h = (int)CGImageGetHeight(cg);
        if (w > 0 && h > 0) {
          // CGBitmapContext cannot emit 24-bit BGR, so draw BGRA and drop alpha
          // with cvtColor — still one CPU pass over the page, against a decode
          // that ran on the media engine.
          cv::Mat bgra(h, w, CV_8UC4);
          CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
          CGContextRef ctx = CGBitmapContextCreate(
              bgra.data, (size_t)w, (size_t)h, 8, bgra.step, cs,
              kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
          CGColorSpaceRelease(cs);
          if (ctx) {
            CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), cg);
            CGContextRelease(ctx);
            CGImageRelease(cg);
            cv::Mat bgr;
            cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
            decoded_ = std::make_unique<MetalImage>(MetalImage::from_host_bgr(bgr));
            return decoded_->view();
          }
        }
        CGImageRelease(cg);
      }
    }
  }

  cv::Mat enc(1, (int)len, CV_8U, const_cast<std::uint8_t *>(data));
  cv::Mat bgr = cv::imdecode(enc, cv::IMREAD_COLOR);
  if (bgr.empty()) return {};
  decoded_ = std::make_unique<MetalImage>(MetalImage::from_host_bgr(bgr));
  return decoded_->view();
}

void MetalKernels::resize_normalize(const backend::ImageView &src, float *dst_chw,
                                    int dst_w, int dst_h,
                                    const backend::NormParams &params,
                                    backend::DeviceQueue &queue) {
  // On ANY failure blank the destination. IKernels::resize_normalize has no
  // return channel, and dst_chw is stage scratch reused for every page — an
  // early `return` would leave the PREVIOUS page's normalized canvas there for
  // the model to run on, producing confident boxes for the wrong image.
  //
  // A CONTRACT REFUSAL IS ONE OF THOSE FAILURES: kernels.h says a refusing void
  // op "returns without writing", which is the requirement not to substitute
  // different pixels — it is not permission to leave the LAST page's pixels in
  // place, which is a silent substitution of the worst kind. So the guard below
  // blanks too, exactly like the texture/buffer failures further down.
  const auto blank_dst = [&] {
    std::memset(dst_chw, 0, (std::size_t)dst_w * dst_h * 3 * sizeof(float));
  };
  // PARAMETER CONTRACT (kernels.h): NormPath::FullFrame is what makes
  // caps().params.norm_mean_std_full_frame load-bearing for this op. This
  // backend's shader IS fully parameterized, so the flag stays true and nothing
  // is refused here that was not refused before.
  if (!backend::require_norm_supported(params, caps().params,
                                       "MetalKernels::resize_normalize",
                                       backend::NormPath::FullFrame)) {
    blank_dst();
    return;
  }
  @autoreleasepool {
    auto &mq = as_metal(queue);
    MPSCommandBuffer *cb = mq.acquire_cb();
    // The page pack (when needed) is encoded onto THIS command buffer, ahead of
    // the resize below — see ensure_texture().
    id<MTLTexture> tex = ensure_texture(src, cb);
    if (!tex) { NSLog(@"[apple] resize_normalize: no source texture"); blank_dst(); return; }
    std::size_t off = 0;
    id<MTLBuffer> dst = resolve_buffer(dst_chw, &off);
    if (!dst) { NSLog(@"[apple] resize_normalize: dst not a Metal buffer"); blank_dst(); return; }

    RNParams p{};
    p.dst_w = (std::uint32_t)dst_w; p.dst_h = (std::uint32_t)dst_h;
    p.src_w = (std::uint32_t)src.cols; p.src_h = (std::uint32_t)src.rows;
    p.mean0 = params.mean[0]; p.mean1 = params.mean[1]; p.mean2 = params.mean[2];
    p.istd0 = params.inv_std[0]; p.istd1 = params.inv_std[1]; p.istd2 = params.inv_std[2];
    p.order = params.order == backend::ChannelOrder::BGR ? 0u : 1u;
    p.letterbox = params.letterbox ? 1u : 0u;

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:mtl_pipeline("resize_normalize")];
    [enc setTexture:tex atIndex:0];
    [enc setBuffer:dst offset:off atIndex:0];
    [enc setBytes:&p length:sizeof(p) atIndex:1];
    [enc dispatchThreads:MTLSizeMake(dst_w, dst_h, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
    mq.submit_cb(cb);
  }
}

void MetalKernels::warp_crops(const backend::ImageView &src, const float *d_M_invs,
                              const int *d_crop_widths, float *d_dst_batch,
                              int batch_size, int dst_h, int dst_w,
                              const backend::NormParams &params,
                              backend::DeviceQueue &queue) {
  // The proven fused warp (tools/probes/apple/warp.metal): sample the page texture through the
  // per-crop inverse homography, write planar NCHW in [-1,1]. Normalization is
  // BAKED into the shader (rec's (v/127.5 - 1)).
  //
  // PARAMETER CONTRACT (kernels.h): this op can therefore serve exactly one
  // distribution. Match it and refuse anything else, rather than accepting a
  // NormParams and quietly ignoring it. rec and cls both want rec_norm(), so
  // no live caller is affected — but an ImageNet cls call (the bug three other
  // backends shipped) now fails loudly here instead of being absorbed.
  //
  // The whitelist below is only the POSITIVE match; the refusal itself is the
  // SHARED guard, driven by caps().params.norm_mean_std_warp (declared false
  // above). Hand-rolling the refusal here left that flag unread, which is how a
  // future backend could declare the same limitation and still substitute
  // silently.
  //
  // Same blanking reasoning as resize_normalize: the crop batch is reused
  // scratch, so a bare `return` — refusal included — would feed the recognizer
  // the PREVIOUS page's crops.
  const auto blank_dst = [&] {
    std::memset(d_dst_batch, 0,
                (std::size_t)batch_size * 3 * dst_h * dst_w * sizeof(float));
  };
  if (!backend::norm_equal(params, backend::norm::rec_norm())) {
    (void)backend::refuse_unbaked_norm(
        params, caps().params, "MetalKernels::warp_crops",
        backend::NormPath::Warp,
        "NormParams other than norm::rec_norm() (warp.metal bakes it in)");
    blank_dst();
    return;
  }
  @autoreleasepool {
    auto &mq = as_metal(queue);
    MPSCommandBuffer *cb = mq.acquire_cb();
    id<MTLTexture> tex = ensure_texture(src, cb);
    if (!tex) { NSLog(@"[apple] warp_crops: no source texture"); blank_dst(); return; }
    std::size_t o0 = 0, o1 = 0, o2 = 0;
    id<MTLBuffer> dst = resolve_buffer(d_dst_batch, &o0);
    id<MTLBuffer> Hb = resolve_buffer(d_M_invs, &o1);
    id<MTLBuffer> cw = resolve_buffer(d_crop_widths, &o2);
    if (!dst || !Hb || !cw) { NSLog(@"[apple] warp_crops: buffer resolve failed"); blank_dst(); return; }

    std::uint32_t dims[4] = {(std::uint32_t)batch_size, 3u, (std::uint32_t)dst_h,
                             (std::uint32_t)dst_w};

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:mtl_pipeline("warp_crops")];
    [enc setTexture:tex atIndex:0];
    [enc setBuffer:dst offset:o0 atIndex:0];
    [enc setBuffer:Hb offset:o1 atIndex:1];
    [enc setBytes:dims length:sizeof(dims) atIndex:2];
    [enc setBuffer:cw offset:o2 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(dst_w, dst_h, batch_size)
        threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
    [enc endEncoding];
    mq.submit_cb(cb);
  }
}

void MetalKernels::threshold(const float *src, std::uint8_t *dst, int w, int h,
                             int batch_size, float thresh,
                             backend::DeviceQueue &queue) {
  @autoreleasepool {
    std::size_t so = 0, dofb = 0;
    id<MTLBuffer> sb = resolve_buffer(src, &so);
    id<MTLBuffer> db = resolve_buffer(dst, &dofb);
    if (!sb || !db) { NSLog(@"[apple] threshold: buffer resolve failed"); return; }
    std::uint32_t n = (std::uint32_t)batch_size * (std::uint32_t)w * (std::uint32_t)h;

    auto &mq = as_metal(queue);
    MPSCommandBuffer *cb = mq.acquire_cb();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:mtl_pipeline("threshold_u8")];
    [enc setBuffer:sb offset:so atIndex:0];
    [enc setBuffer:db offset:dofb atIndex:1];
    [enc setBytes:&thresh length:sizeof(float) atIndex:2];
    [enc setBytes:&n length:sizeof(std::uint32_t) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    mq.submit_cb(cb);
  }
}

std::vector<turbo_ocr::Box>
MetalKernels::db_postprocess(const float *d_pred_map, const std::uint8_t *d_bitmap,
                             int w, int h, const backend::DbPostParams &params,
                             backend::DeviceQueue &queue) {
  // PARAMETER CONTRACT (kernels.h): honour or refuse loudly, never substitute.
  if (!backend::require_db_supported(params, caps().params,
                                     "MetalKernels::db_postprocess"))
    return {};
  // HOST FALLBACK. Unified memory: d_pred_map / d_bitmap .contents are readable
  // on the host directly, so this is a coherent read (not a PCIe D2H). Ensure the
  // GPU threshold/inference that produced them is done first.
  queue.synchronize();
  cv::Mat pred(h, w, CV_32F, const_cast<float *>(d_pred_map));
  cv::Mat bitmap(h, w, CV_8U, const_cast<std::uint8_t *>(d_bitmap));

  // Interface contract: boxes are returned in the MAP's (resized) coordinate
  // space; the caller rescales to original dims. So pass orig == resize == (w,h).
  std::vector<cv::Point> shifted_buf;
  cv::Mat mask_buf;
  std::vector<std::vector<cv::Point>> contours_buf;
  std::vector<cv::Vec4i> hier_buf;
  return turbo_ocr::detection::extract_boxes_from_bitmap(
      pred, bitmap, /*orig_h*/ h, /*orig_w*/ w, /*resize_h*/ h, /*resize_w*/ w,
      params.box_thresh, params.unclip_ratio, params.min_box_side,
      // WAS 2.0f — every other call site (host_kernels.cpp, sycl_kernels.cpp,
      // and MpsDetector's own db_postprocess_ in this same backend) passes 5.0,
      // so Apple was internally inconsistent with its own detector. The value
      // now comes from DbPostParams, which defaults from
      // detection::kMinUnclippedSide.
      params.min_unclipped_side, shifted_buf, mask_buf, contours_buf, hier_buf);
}

void MetalKernels::argmax(const float *input_probs, int *output_indices,
                          float *output_scores, int batch_size, int seq_len,
                          int num_classes, backend::DeviceQueue &queue) {
  @autoreleasepool {
    std::size_t io = 0, oi = 0, os = 0;
    id<MTLBuffer> lb = resolve_buffer(input_probs, &io);
    id<MTLBuffer> ib = resolve_buffer(output_indices, &oi);
    id<MTLBuffer> mb = resolve_buffer(output_scores, &os);
    if (!lb || !ib || !mb) { NSLog(@"[apple] argmax: buffer resolve failed"); return; }
    std::uint32_t dims[3] = {(std::uint32_t)batch_size, (std::uint32_t)seq_len,
                             (std::uint32_t)num_classes};

    auto &mq = as_metal(queue);
    MPSCommandBuffer *cb = mq.acquire_cb();
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:mtl_pipeline("argmax")];
    [enc setBuffer:lb offset:io atIndex:0];
    [enc setBuffer:ib offset:oi atIndex:1];
    [enc setBuffer:mb offset:os atIndex:2];
    [enc setBytes:dims length:sizeof(dims) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(seq_len, batch_size, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    mq.submit_cb(cb);
  }
}

void MetalKernels::preprocess_region(const backend::ImageView & /*src*/,
                                     const backend::Rect & /*rect*/,
                                     backend::PreprocKind /*kind*/,
                                     float * /*dst_chw*/,
                                     backend::DeviceQueue & /*queue*/) {
  // TODO(apple-fused-region-preproc): fused table/layout region preprocessors (LayoutSubRect / TableCls /
  // SlanextBGR / SlanextRGB). Table/layout stages are not yet resident on Apple
  // (see mps_stages.h). caps().preprocess_region == false signals the fallback.
  static bool warned = false;
  if (!warned) {
    NSLog(@"[apple] preprocess_region not implemented "
          "(TODO(apple-fused-region-preproc))");
    warned = true;
  }
}

} // namespace turbo_ocr::apple
