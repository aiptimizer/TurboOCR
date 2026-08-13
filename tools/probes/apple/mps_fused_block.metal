#include <metal_stdlib>
using namespace metal;

// erf via Abramowitz-Stegun 7.1.26 (max abs err ~1.5e-7) — Metal has no builtin erf here.
inline float erf_as(float x){
    float s = x<0.0f ? -1.0f : 1.0f; float ax=fabs(x);
    float t = 1.0f/(1.0f+0.3275911f*ax);
    float y = 1.0f - (((((1.061405429f*t - 1.453152027f)*t) + 1.421413741f)*t - 0.284496736f)*t + 0.254829592f)*t*exp(-ax*ax);
    return s*y;
}

// Fused depthwise(3x3,pad1)+bias -> pointwise(48->96,1x1)+bias -> GELU.
// One thread = one output pixel of one crop; the 48 depthwise results live in
// registers and are consumed by the pointwise dot products WITHOUT ever being
// written to global memory. This is the whole point: kill the inter-layer
// activation round-trip that makes MPSGraph memory-bound on this net.
kernel void fused_dwsep(
    device const float* in   [[buffer(0)]],   // [B,48,12,80] NCHW
    device const float* wdw  [[buffer(1)]],   // [48,9]
    device const float* bdw  [[buffer(2)]],   // [48]
    device const float* wpw  [[buffer(3)]],   // [96,48]
    device const float* bpw  [[buffer(4)]],   // [96]
    device float*       out  [[buffer(5)]],   // [B,96,12,80]
    constant uint&      B    [[buffer(6)]],
    threadgroup float*  twpw [[threadgroup(0)]],  // pointwise weights cached [96*48]
    uint3 gid [[thread_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]])
{
    const uint W=80, H=12, Cin=48, Cout=96, TGN=64;  // threadgroup = 16x4
    // cooperatively cache the 96x48 pointwise weights in threadgroup memory (loaded
    // once per threadgroup, reused across all its pixels) instead of per-thread global reads
    for(uint i=tid; i<Cout*Cin; i+=TGN) twpw[i]=wpw[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint ow=gid.x, oh=gid.y, b=gid.z;
    if(ow>=W || oh>=H || b>=B) return;

    float dw[Cin];
    for(uint ci=0; ci<Cin; ++ci){
        float acc = bdw[ci];
        for(int dy=0; dy<3; ++dy){
            int iy = int(oh)+dy-1; if(iy<0 || iy>=int(H)) continue;
            uint base = ((b*Cin+ci)*H + uint(iy))*W;
            for(int dx=0; dx<3; ++dx){
                int ix = int(ow)+dx-1; if(ix<0 || ix>=int(W)) continue;
                acc += in[base + uint(ix)] * wdw[ci*9 + dy*3 + dx];
            }
        }
        dw[ci] = acc;            // stays in registers
    }
    for(uint co=0; co<Cout; ++co){
        float acc = bpw[co];
        uint wb = co*Cin;
        for(uint ci=0; ci<Cin; ++ci) acc += dw[ci]*twpw[wb+ci];
        float g = acc*0.5f*(1.0f + erf_as(acc*0.70710678118f));   // GELU
        out[((b*Cout+co)*H+oh)*W+ow] = g;
    }
}
