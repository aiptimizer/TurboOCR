#include <metal_stdlib>
using namespace metal;

// TurboOCR Apple backend — device pre/post compute kernels.
//
// These implement the IKernels (backend/kernels.h) op set that has a Metal
// primitive. The star is warp_crops (verbatim from tools/probes/apple/warp.metal), the fused
// perspective-warp+resize+normalize that feeds the MPSGraph recognizer with zero
// host round-trips — the residency win the POC measured (tools/probes/apple/mps_ocr.mm:140).
// resize_normalize, argmax, threshold and pack_bgr8_to_rgba round out the set.
// db_postprocess (CCL + unclip) has no Metal primitive and is a HOST fallback in
// MetalKernels (see metal_kernels.mm) until a Metal union-find is hand-written.
//
// MEMORY CONVENTION (mirrors kernels.h): every buffer argument is a caller-owned
// device (MTLBuffer.contents) allocation in the queue's Metal space; no kernel
// allocates scratch. Textures carry the source page (hardware bilinear sampling).

// ---------------------------------------------------------------------------
// Fused perspective-warp + bilinear-resize + normalize  (from tools/probes/apple/warp.metal)
// ---------------------------------------------------------------------------
// One thread per output pixel of one crop; all B crops in the z-grid dim. Reads
// the source page from a texture (hardware bilinear), writes planar NCHW fp32 in
// [-1,1] — the exact layout rec_tiny expects — so the MPSGraph recognizer can
// consume this MTLBuffer directly. H holds the 9-float row-major inverse
// homography (dst-pixel -> src-pixel) per crop, flat to dodge float3x3 alignment.
kernel void warp_crops(
    texture2d<float, access::sample> src [[texture(0)]],
    device float*        dst      [[buffer(0)]],   // [B,3,H,W]
    device const float*  H        [[buffer(1)]],   // 9 floats per crop
    constant uint4&      dims     [[buffer(2)]],   // (B, 3, H, W)
    device const int*    contentW [[buffer(3)]],   // per-crop content width (< W => pad)
    uint3 gid [[thread_position_in_grid]])
{
    const uint B = dims.x, Ht = dims.z, Wt = dims.w;
    if (gid.x >= Wt || gid.y >= Ht || gid.z >= B) return;

    const uint plane = Ht * Wt;
    const uint base  = gid.z * 3u * plane + gid.y * Wt + gid.x;

    // Columns past the crop's natural content width are zero (mid-gray in [-1,1]),
    // matching the rec model's right-padding — one static [B,3,H,W] batch shape.
    if (gid.x >= (uint)contentW[gid.z]) {
        dst[base] = 0.0f; dst[base + plane] = 0.0f; dst[base + 2u*plane] = 0.0f;
        return;
    }

    const uint o = gid.z * 9u;
    const float px = float(gid.x) + 0.5f;
    const float py = float(gid.y) + 0.5f;
    const float sx = H[o+0]*px + H[o+1]*py + H[o+2];
    const float sy = H[o+3]*px + H[o+4]*py + H[o+5];
    const float sw = H[o+6]*px + H[o+7]*py + H[o+8];

    constexpr sampler s(coord::pixel, filter::linear, address::clamp_to_edge);
    float3 rgb = src.sample(s, float2(sx / sw, sy / sw)).rgb;  // 0..1
    float3 n   = rgb * 2.0f - 1.0f;                            // -> [-1,1] (v/127.5 - 1)

    dst[base + 0u*plane] = n.x;
    dst[base + 1u*plane] = n.y;
    dst[base + 2u*plane] = n.z;
}

// ---------------------------------------------------------------------------
// resize_normalize — resize a source texture into a normalized CHW fp32 tensor.
// ---------------------------------------------------------------------------
// One thread per output pixel. The source texture is sampled in [0,1] (the
// sampler already divides by 255), so the normalization is out = (v - mean)*inv_std
// operating on that [0,1] value — inv_scale (1/255) is folded into the texture.
// Generalizes cuda_fused_resize_normalize_det / _layout (backend/kernels.h).
struct RNParams {
    uint  dst_w, dst_h;
    uint  src_w, src_h;
    float mean0, mean1, mean2;      // indexed in `order`
    float istd0, istd1, istd2;
    uint  order;                    // 0 = BGR planes, 1 = RGB planes
    uint  letterbox;                // 0 = stretch, 1 = preserve-AR + pad
    // Snapped-canvas content region (0,0 = disabled): resize the page into the
    // TOP-LEFT content_w x content_h rectangle and write 0 in NORMALIZED space
    // (= the per-channel mean pixel, which DB scores as background) everywhere
    // else — the shared detection::snap_det_canvas_grid letterbox policy the
    // Intel backend repacks on the host; here it is one kernel.
    uint  content_w, content_h;
};

kernel void resize_normalize(
    texture2d<float, access::sample> src [[texture(0)]],
    device float*        dst    [[buffer(0)]],   // [3, dst_h, dst_w]
    constant RNParams&   p      [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.dst_w || gid.y >= p.dst_h) return;
    const uint plane = p.dst_h * p.dst_w;
    const uint idx   = gid.y * p.dst_w + gid.x;

    float3 rgb;
    if (p.content_w != 0u && p.content_h != 0u) {
        // Content-region mode: pad pixels are 0 in NORMALIZED space, so they are
        // written directly, BEFORE the mean/std normalization below.
        if (gid.x >= p.content_w || gid.y >= p.content_h) {
            dst[idx + 0u*plane] = 0.0f;
            dst[idx + 1u*plane] = 0.0f;
            dst[idx + 2u*plane] = 0.0f;
            return;
        }
        constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
        const float u = (float(gid.x) + 0.5f) / float(p.content_w);
        const float v = (float(gid.y) + 0.5f) / float(p.content_h);
        rgb = src.sample(s, float2(u, v)).rgb;
    } else if (p.letterbox) {
        // Preserve aspect ratio, center on a padded canvas (0-fill = black).
        const float scale = min(float(p.dst_w) / float(p.src_w),
                                float(p.dst_h) / float(p.src_h));
        const float nw = float(p.src_w) * scale, nh = float(p.src_h) * scale;
        const float ox = (float(p.dst_w) - nw) * 0.5f, oy = (float(p.dst_h) - nh) * 0.5f;
        const float fx = (float(gid.x) - ox), fy = (float(gid.y) - oy);
        if (fx < 0.0f || fy < 0.0f || fx >= nw || fy >= nh) {
            rgb = float3(0.0f);
        } else {
            constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
            rgb = src.sample(s, float2((fx + 0.5f) / nw, (fy + 0.5f) / nh)).rgb;
        }
    } else {
        constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
        const float u = (float(gid.x) + 0.5f) / float(p.dst_w);
        const float v = (float(gid.y) + 0.5f) / float(p.dst_h);
        rgb = src.sample(s, float2(u, v)).rgb;   // texture is RGB (packed R,G,B)
    }

    // Emit planes in the requested channel order. The source texture is packed
    // R,G,B; BGR order (det/PaddleOCR) writes plane0=B, plane1=G, plane2=R with
    // mean/inv_std indexed 0,1,2 onto those planes.
    float3 ch = (p.order == 0u) ? float3(rgb.b, rgb.g, rgb.r) : rgb;
    dst[idx + 0u*plane] = (ch.x - p.mean0) * p.istd0;
    dst[idx + 1u*plane] = (ch.y - p.mean1) * p.istd1;
    dst[idx + 2u*plane] = (ch.z - p.mean2) * p.istd2;
}

// ---------------------------------------------------------------------------
// pack_bgr8_to_rgba — device BGR8 buffer -> RGBA8 texture (for warp sampling).
// ---------------------------------------------------------------------------
// Canonical decode format is interleaved 8-bit BGR (ImageView contract). The
// warp/resize kernels sample an RGBA8 texture whose .rgb is R,G,B, so this pack
// swaps B<->R while promoting to 4 channels — keeping the page resident (no host
// round-trip) exactly as tools/probes/apple/mps_ocr.mm:66-69 did on the host at upload time.
kernel void pack_bgr8_to_rgba(
    device const uchar*  bgr   [[buffer(0)]],
    constant uint3&      dims  [[buffer(1)]],   // (w, h, row_step_bytes)
    texture2d<float, access::write> tex [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint w = dims.x, h = dims.y, step = dims.z;
    if (gid.x >= w || gid.y >= h) return;
    const uint o = gid.y * step + gid.x * 3u;
    const float inv = 1.0f / 255.0f;
    tex.write(float4(float(bgr[o + 2u]) * inv,   // R
                     float(bgr[o + 1u]) * inv,   // G
                     float(bgr[o + 0u]) * inv,   // B
                     1.0f),
              gid);
}

// ---------------------------------------------------------------------------
// argmax — per-timestep argmax + max over the class axis (CTC prep).
// ---------------------------------------------------------------------------
// For each [b, t] position over num_classes, write the winning class index and
// its score. Generalizes cuda_argmax. Used when a caller holds raw [B,T,C] logits
// in a buffer; the recognizer's default path instead folds argmax into the
// MPSGraph (reductionArgMaximum) so only [B,T] indices cross to host — see
// mps_engine.mm / tools/probes/apple/mps_ocr.mm:119.
kernel void argmax(
    device const float* logits [[buffer(0)]],   // [B, T, C]
    device int*         out_idx [[buffer(1)]],  // [B, T]
    device float*       out_max [[buffer(2)]],  // [B, T]
    constant uint3&     dims    [[buffer(3)]],   // (B, T, C)
    uint2 gid [[thread_position_in_grid]])       // (t, b)
{
    const uint T = dims.y, C = dims.z;
    if (gid.y >= dims.x || gid.x >= T) return;
    const uint base = (gid.y * T + gid.x) * C;
    float best = logits[base]; int bi = 0;
    for (uint c = 1; c < C; ++c) {
        float v = logits[base + c];
        if (v > best) { best = v; bi = int(c); }
    }
    const uint p = gid.y * T + gid.x;
    out_idx[p] = bi;
    out_max[p] = best;
}

// ---------------------------------------------------------------------------
// threshold — float probability map -> uint8 bitmap (255=fg, 0=bg).
// ---------------------------------------------------------------------------
// Over batch*w*h elements. Generalizes cuda_threshold_to_u8 / _batch. The DB
// detector thresholds the prob map before host CCL (see metal_kernels.mm).
kernel void threshold_u8(
    device const float* src  [[buffer(0)]],
    device uchar*       dst  [[buffer(1)]],
    constant float&     thr  [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = src[gid] > thr ? 255 : 0;
}
