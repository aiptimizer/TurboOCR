#include <metal_stdlib>
using namespace metal;

// Fused perspective-warp + bilinear-resize + normalize.
// One thread per output pixel of one crop; all N crops in the z-grid dim.
// Reads the source page from a texture (hardware bilinear), writes planar
// NCHW fp32 in [-1,1] — the exact layout rec_tiny expects — so the MPSGraph
// recognizer can consume this MTLBuffer with zero host round-trips.
//
// H holds 9 floats per crop (row-major inverse homography: dst-pixel -> src-
// pixel), passed flat to dodge float3x3 buffer-alignment pitfalls.
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
