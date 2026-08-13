// Bounded per-component Euclidean unclip (DBNet-style polygon offset), computed
// as a boundary-seeded disc dilation rather than a global distance transform.
//
// The unclip only ever expands a component outward by a few pixels
// (area*ratio/perimeter, clamped to [min_expand, max_expand]). Computing a
// GLOBAL distance transform (JFA over log2(dim) full-grid passes, plus a
// JFA+1 pass that is still only an APPROXIMATE nearest-seed) to answer that
// purely local, small-radius question is wasteful. Instead, seed from the
// component boundary only:
//   1. The nearest foreground pixel to any background pixel is always a
//      foreground BOUNDARY pixel (a foreground pixel with a background
//      neighbour) — an interior foreground pixel always has a strictly closer
//      neighbour toward the query. So only boundary pixels need to be seeds.
//   2. Each boundary seed "stamps" the exact squared distance + its component
//      label into every pixel inside ITS OWN expand radius, resolving ties by
//      an atomicMin over the packed (distance,label) key.
//   3. Foreground pixels keep their own label directly.
//   4. Atomic-scatter bbox extraction (unchanged).
//
// SCOPE OF "exact": the squared distance and winner selection are exact (no JFA
// boundary error), but the winner is the nearest REACHING component — the
// nearest one whose OWN expand radius covers this pixel — NOT the global-Voronoi
// nearest component. Two components with unequal radii differ exactly at their
// shared border (a pixel nearer to A with radius 2 but only reached by B with
// radius 6 is labelled B). This is deliberate: it matches the CPU reference,
// which offsets each contour INDEPENDENTLY (src/analysis/detection/det_postprocess.cpp) —
// it is not a true Voronoi partition. The expanded region is a union of radius-r
// discs (a morphological disc-dilation); its rounded corners differ from a
// polygon offset's mitre/round joins, but the only consumer here reduces it to
// an axis-aligned bbox, which washes that out. The remaining accuracy ceiling is
// the per-component radius itself: compute_expand_per_comp uses a BBOX-rectangle
// area/perimeter, exact only for axis-aligned rectangles (see that kernel).
//
// Work is proportional to (#boundary pixels) * (per-component radius^2): each
// boundary seed stamps a (2r+1)^2 neighbourhood, so adjacent seeds' stamps
// overlap and the atomicMin count is ~(2r+1)^2 per boundary pixel — cheaper than
// full-grid JFA at DBNet radii (r=2-4), but it scales quadratically in r, so a
// large max_expand would degrade faster than JFA (kMaxSafeExpand guards the key
// budget, NOT this runtime).

#include "nvidia/kernels_cuda/kernels_cuda.h"
#include "nvidia/support/cuda_check.h"
#include <cuda_runtime.h>
#include <climits>
#include <cmath>

namespace turbo_ocr::kernels {

// Per-pixel winner key: squared distance in the high bits, component id in the
// low kLabelBits — so a single unsigned atomicMin picks the nearest seed and
// carries its label with no coordinate lookup. 0xFFFFFFFF = unclaimed.
// kLabelBits=13 holds up to 8191 components (> kMaxGpuComponents) and the max
// stamped d2 (<= max_expand^2, a few hundred) shifts well within 32 bits.
static constexpr int kLabelBits = 13;
static constexpr uint32_t kLabelMask = (1u << kLabelBits) - 1u;
static constexpr uint32_t kKeyEmpty = 0xFFFFFFFFu;
static_assert(kMaxGpuComponents <= (int)kLabelMask, "label id must fit in key");

// Distance budget: pack_key shifts d2 left kLabelBits, so d2 must stay below
// 2^(32-kLabelBits). Each seed's radius is clamped to kMaxSafeExpand so a
// misconfigured max_expand can never overflow the key. DBNet radii are 2-4 px;
// this only guards the pathological-config case.
static constexpr int kMaxSafeExpand = 723;
static_assert(kMaxSafeExpand * kMaxSafeExpand < (1 << (32 - kLabelBits)),
              "seed radius must keep squared distance within the key budget");

__device__ __forceinline__ uint32_t pack_key(int d2, int cid) {
    return ((uint32_t)d2 << kLabelBits) | (uint32_t)cid;
}

// One thread per pixel. Foreground boundary pixels (a foreground pixel with an
// in-bounds background neighbour) scatter their exact-distance/label stamp into
// every pixel within their own component's expand radius. Non-seed threads exit
// immediately, so the launch cost is dominated by the sparse text boundary.
__global__ void unclip_scatter_kernel(uint32_t *best,
                                       const uint8_t *bitmap,
                                       const int32_t *compact_ids,
                                       const float *expand_per_comp,
                                       int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    if (!bitmap[idx]) return; // background: never a seed

    // Boundary test (8-connectivity): keep only foreground pixels that touch
    // background. Interior pixels can never be the nearest seed of a bg pixel.
    bool boundary = false;
    #pragma unroll
    for (int dy = -1; dy <= 1 && !boundary; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if ((dx | dy) == 0) continue;
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            if (!bitmap[ny * w + nx]) { boundary = true; break; }
        }
    }
    if (!boundary) return;

    int cid = compact_ids[idx];
    if (cid < 0) return;
    float e = expand_per_comp[cid]; // 0 for filtered-out / rejected components
    if (e <= 0.0f) return;
    if (e > (float)kMaxSafeExpand) e = (float)kMaxSafeExpand; // key-budget guard
    float e2 = e * e;
    int r = (int)ceilf(e);

    for (int dy = -r; dy <= r; ++dy) {
        int ty = y + dy;
        if (ty < 0 || ty >= h) continue;
        int dy2 = dy * dy;
        for (int dx = -r; dx <= r; ++dx) {
            int d2 = dx * dx + dy2;
            // Same cutoff as the resolve step's radius test: bit-for-bit the
            // effective per-component expand distance as before.
            if ((float)d2 > e2) continue;
            int tx = x + dx;
            if (tx < 0 || tx >= w) continue;
            atomicMin(&best[ty * w + tx], pack_key(d2, cid));
        }
    }
}

// Resolve each pixel to its final compact label (1..N, 0 = background).
// Foreground pixels take their own component; background pixels take the
// winning seed recorded during scatter (already guaranteed within radius).
__global__ void unclip_resolve_kernel(uint32_t *expanded_labels,
                                       const uint32_t *best,
                                       const uint8_t *bitmap,
                                       const int32_t *compact_ids,
                                       const float *expand_per_comp,
                                       int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;

    if (bitmap[idx]) {
        int cid = compact_ids[idx];
        if (cid < 0 || expand_per_comp[cid] <= 0.0f) { expanded_labels[idx] = 0; return; }
        expanded_labels[idx] = (uint32_t)(cid + 1);
        return;
    }
    uint32_t key = best[idx];
    if (key == kKeyEmpty) { expanded_labels[idx] = 0; return; }
    int cid = (int)(key & kLabelMask);
    expanded_labels[idx] = (uint32_t)(cid + 1);
}

// One thread per pixel: accumulate each component's CONTOUR PERIMETER as the
// count of exposed 4-connectivity crack edges (a foreground pixel edge that
// faces background or the image border). For a solid axis-aligned W x H
// component the crack count is exactly 2*(W+H) == the bbox perimeter, so the
// common horizontal-text case is unchanged; rotated / ragged components get a
// staircase-length perimeter instead of the bbox rectangle's.
//
// CONVENTION NOTE: this is the pixel-COUNT convention, deliberately paired
// with area = pixel_count in compute_expand_per_comp. The CPU reference
// (cv2.contourArea / cv2.arcLength on pixel-CENTRE polygons) measures the same
// box as (W-1)(H-1) over 2(W+H)-4 — both numerator and denominator smaller —
// so the two expand distances differ by up to a few percent on SMALL boxes
// (e.g. 100x20: 8.33 vs 7.97 px) and converge for large ones; the [min,max]
// expand clamp masks most of the residual. One consistent convention is used
// here; do NOT mix crack perimeter with contour area or vice versa.
// perim_per_comp MUST be zeroed before launch. Indexed by PRE-filter compact_id.
__global__ void accumulate_crack_perimeter_kernel(
    const int32_t *compact_ids, const uint8_t *bitmap,
    int w, int h, int *perim_per_comp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    if (!bitmap[idx]) return;
    int cid = compact_ids[idx];
    if (cid < 0) return;
    int cracks = 0;
    if (x == 0 || !bitmap[idx - 1]) ++cracks;
    if (x == w - 1 || !bitmap[idx + 1]) ++cracks;
    if (y == 0 || !bitmap[idx - w]) ++cracks;
    if (y == h - 1 || !bitmap[idx + w]) ++cracks;
    if (cracks) atomicAdd(&perim_per_comp[cid], cracks);
}

// Per-component expand distance for ALL pre-filter compact_ids.
// expand_i = area_i * ratio / perimeter_i, indexed by PRE-filter compact_id so
// the seed lookup is consistent. The perimeter is the per-component contour
// (crack) perimeter accumulated above, matching cv2.arcLength; it falls back to
// the bbox perimeter only if the accumulation produced nothing (never for a
// pc>=3 component). Returns 0 for empty / size-rejected / score-rejected slots.
__global__ void compute_expand_per_comp_kernel(
    const GpuDetBox *bboxes, const int *perim_per_comp, int num_slots,
    float unclip_ratio, float min_expand, float max_expand,
    float box_thresh, float *expand_per_comp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_slots) return;
    const GpuDetBox &b = bboxes[i];
    int pc = b.pixel_count;
    if (pc < 3) { expand_per_comp[i] = 0.0f; return; }
    int bw = b.xmax - b.xmin + 1;
    int bh = b.ymax - b.ymin + 1;
    if (bw < 3 || bh < 3) { expand_per_comp[i] = 0.0f; return; }
    float score = b.score / (float)pc;
    if (score < box_thresh) { expand_per_comp[i] = 0.0f; return; }
    float perim = (float)perim_per_comp[i];
    if (perim < 1.0f) perim = 2.0f * (float)(bw + bh);
    float area = (float)pc;
    float e = area * unclip_ratio / perim;
    if (e < min_expand) e = min_expand;
    if (e > max_expand) e = max_expand;
    expand_per_comp[i] = e;
}

// Step 4a: init bbox slots with sentinels so atomic scatter min/max works.
__global__ void jfa_init_bboxes_kernel(GpuDetBox *bboxes, int num_slots) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_slots) return;
    GpuDetBox &b = bboxes[i];
    b.xmin = INT_MAX; b.ymin = INT_MAX;
    b.xmax = INT_MIN; b.ymax = INT_MIN;
    b.pixel_count = 0;
    b.score = 0.0f;
}

// Step 4b: image-wide single pass, atomic scatter. One thread per pixel
// atomic-updates the bbox slot for its label.
__global__ void jfa_extract_bboxes_kernel(
    const uint32_t *expanded_labels,
    int w, int h,
    GpuDetBox *bboxes, int num_slots) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = w * h;
    if (idx >= total) return;
    uint32_t label = expanded_labels[idx];
    if (label == 0) return;
    int cid = (int)label - 1;
    if (cid >= num_slots) return;
    int x = idx % w, y = idx / w;
    GpuDetBox &b = bboxes[cid];
    atomicMin(&b.xmin, x); atomicMax(&b.xmax, x);
    atomicMin(&b.ymin, y); atomicMax(&b.ymax, y);
    atomicAdd(&b.pixel_count, 1);
}

// ---------------------------------------------------------------------------
// Oriented (rotated) min-area-rect via PCA over the expanded region.
//
// cv2.minAreaRect is rotating-calipers on the convex hull (exact min area).
// PCA minimizes the SECOND MOMENT, not the bounding area, so the two agree only
// for a uniform-density rectangle; on real text blobs (ascenders/descenders,
// ragged ends, uneven stroke density) the PCA angle differs from minAreaRect's
// by typically a degree or two — negligible for crop-and-recognize, and measured
// at parity with the CPU reference on skewed text. Two known residuals: (1)
// near-isotropic blobs, where the PCA angle is ill-conditioned — guarded below
// by an anisotropy fallback to axis-aligned; (2) the projection extents below are
// taken over the disc-DILATED region, so a rotated component's rounded disc
// corners slightly enlarge the rect vs a mitre/round polygon offset (the same
// disc-corner residual as the bbox path, no longer washed out since orientation
// is kept — small, and validated within noise on skewed OmniDocBench pages). PCA
// is chosen because it is a two-pass atomic reduction (GPU-friendly, no per-
// component hull construction), which is what the sibling detector used to reach
// official parity.
//
// Layout: mom[cid*6 + {0..5}] = {n, sx, sy, sxx, syy, sxy} (uint64).
//         orient[cid*6 + {0..5}] = {cos, sin, umin, umax, vmin, vmax} (float).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float atomic_min_f(float *addr, float val) {
    int *ai = (int *)addr;
    int old = *ai, assumed;
    do {
        assumed = old;
        old = atomicCAS(ai, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float atomic_max_f(float *addr, float val) {
    int *ai = (int *)addr;
    int old = *ai, assumed;
    do {
        assumed = old;
        old = atomicCAS(ai, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Pass A: one thread per expanded-label pixel — axis-aligned bbox (same atomics
// as jfa_extract_bboxes) plus the integer second-moment accumulation for PCA.
__global__ void oriented_extract_kernel(
    const uint32_t *expanded_labels, int w, int h,
    GpuDetBox *bboxes, unsigned long long *mom, int num_slots) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = w * h;
    if (idx >= total) return;
    uint32_t label = expanded_labels[idx];
    if (label == 0) return;
    int cid = (int)label - 1;
    if (cid >= num_slots) return;
    int x = idx % w, y = idx / w;
    GpuDetBox &b = bboxes[cid];
    atomicMin(&b.xmin, x); atomicMax(&b.xmax, x);
    atomicMin(&b.ymin, y); atomicMax(&b.ymax, y);
    atomicAdd(&b.pixel_count, 1);
    unsigned long long *m = mom + (size_t)cid * 6;
    // Widen BEFORE multiplying: int x*x overflows at coords >= 46341. Today's
    // det resize caps coords at 4096, but the cliff would be silent.
    const unsigned long long ux = (unsigned long long)x, uy = (unsigned long long)y;
    atomicAdd(&m[0], 1ULL);
    atomicAdd(&m[1], ux);
    atomicAdd(&m[2], uy);
    atomicAdd(&m[3], ux * ux);
    atomicAdd(&m[4], uy * uy);
    atomicAdd(&m[5], ux * uy);
}

// Pass B (per component): principal-axis angle from the covariance, then seed
// the projection extents. Double math here is safe — one thread per component,
// not a hot per-pixel loop.
__global__ void oriented_axis_kernel(const unsigned long long *mom,
                                     int num_slots, float *orient) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_slots) return;
    const unsigned long long *m = mom + (size_t)i * 6;
    float *o = orient + (size_t)i * 6;
    unsigned long long n = m[0];
    o[2] = INFINITY; o[3] = -INFINITY; o[4] = INFINITY; o[5] = -INFINITY;
    if (n < 3) { o[0] = 1.0f; o[1] = 0.0f; return; } // degenerate → axis-aligned
    double dn = (double)n;
    double mx = (double)m[1] / dn, my = (double)m[2] / dn;
    double cxx = (double)m[3] / dn - mx * mx;
    double cyy = (double)m[4] / dn - my * my;
    double cxy = (double)m[5] / dn - mx * my;
    // Near-isotropic guard: with no dominant axis (cxx≈cyy, cxy≈0) the principal
    // angle is ill-conditioned — atan2(2cxy, cxx-cyy) snaps ~45° on tiny
    // perturbations, so a near-square blob (single glyph, CJK char) would get a
    // spuriously rotated rect where minAreaRect stays axis-aligned. The anisotropy
    // (λmax-λmin)/(λmax+λmin) = sqrt((cxx-cyy)^2 + 4cxy^2)/(cxx+cyy); below a few
    // percent the axis is meaningless, so fall back to axis-aligned (angle 0).
    double trace = cxx + cyy;
    double aniso = sqrt((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy);
    if (trace <= 0.0 || aniso < 0.05 * trace) { o[0] = 1.0f; o[1] = 0.0f; return; }
    double theta = 0.5 * atan2(2.0 * cxy, cxx - cyy);
    o[0] = (float)cos(theta);
    o[1] = (float)sin(theta);
}

// Pass C: one thread per expanded-label pixel — project onto the component's
// oriented basis and track the min/max extent along each axis.
__global__ void oriented_project_kernel(
    const uint32_t *expanded_labels, int w, int h,
    const unsigned long long *mom, float *orient, int num_slots) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = w * h;
    if (idx >= total) return;
    uint32_t label = expanded_labels[idx];
    if (label == 0) return;
    int cid = (int)label - 1;
    if (cid >= num_slots) return;
    if (mom[(size_t)cid * 6] < 3) return;
    int x = idx % w, y = idx / w;
    float *o = orient + (size_t)cid * 6;
    float c = o[0], s = o[1];
    float u = x * c + y * s;
    float v = -x * s + y * c;
    atomic_min_f(&o[2], u); atomic_max_f(&o[3], u);
    atomic_min_f(&o[4], v); atomic_max_f(&o[5], v);
}

// Pass D (per component): reconstruct the 4 oriented-rect corners from the axis
// and the per-axis extents. Degenerate components fall back to the axis-aligned
// bbox (a rotated rect with angle 0).
__global__ void oriented_emit_kernel(const unsigned long long *mom,
                                     const float *orient, int num_slots,
                                     GpuDetBox *bboxes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_slots) return;
    GpuDetBox &b = bboxes[i];
    const float *o = orient + (size_t)i * 6;
    if (mom[(size_t)i * 6] < 3 || o[2] > o[3]) {
        float x0 = (float)b.xmin, y0 = (float)b.ymin;
        float x1 = (float)b.xmax, y1 = (float)b.ymax;
        b.ox[0] = x0; b.oy[0] = y0;
        b.ox[1] = x1; b.oy[1] = y0;
        b.ox[2] = x1; b.oy[2] = y1;
        b.ox[3] = x0; b.oy[3] = y1;
        return;
    }
    float c = o[0], s = o[1];
    float umin = o[2], umax = o[3], vmin = o[4], vmax = o[5];
    // corner(u,v) = u*(c,s) + v*(-s,c)
    const float uu[4] = {umin, umax, umax, umin};
    const float vv[4] = {vmin, vmin, vmax, vmax};
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        b.ox[k] = uu[k] * c - vv[k] * s;
        b.oy[k] = uu[k] * s + vv[k] * c;
    }
}

void cuda_jfa_extract_oriented(const uint32_t *d_expanded_labels,
                               int w, int h,
                               GpuDetBox *d_bboxes, int num_slots,
                               unsigned long long *d_moments, float *d_orient,
                               cudaStream_t stream) {
    if (num_slots <= 0) return;
    int total = w * h;
    int block = 256;
    int comp_grid = (num_slots + block - 1) / block;
    int pix_grid = (total + block - 1) / block;

    jfa_init_bboxes_kernel<<<comp_grid, block, 0, stream>>>(d_bboxes, num_slots);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(d_moments, 0,
                               (size_t)num_slots * 6 * sizeof(unsigned long long), stream));

    oriented_extract_kernel<<<pix_grid, block, 0, stream>>>(
        d_expanded_labels, w, h, d_bboxes, d_moments, num_slots);
    CUDA_CHECK(cudaGetLastError());

    oriented_axis_kernel<<<comp_grid, block, 0, stream>>>(d_moments, num_slots, d_orient);
    CUDA_CHECK(cudaGetLastError());

    oriented_project_kernel<<<pix_grid, block, 0, stream>>>(
        d_expanded_labels, w, h, d_moments, d_orient, num_slots);
    CUDA_CHECK(cudaGetLastError());

    oriented_emit_kernel<<<comp_grid, block, 0, stream>>>(
        d_moments, d_orient, num_slots, d_bboxes);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_jfa_expand_labels(const uint8_t *d_bitmap,
                            const int32_t *d_compact_ids,
                            const float *d_expand_per_comp,
                            uint32_t *d_expanded_labels,
                            int w, int h, float max_expand,
                            uint32_t *d_seeds, uint32_t *d_seeds_alt,
                            cudaStream_t stream) {
    (void)max_expand;    // per-component radius bounds each stamp directly
    (void)d_seeds_alt;   // no ping-pong needed; kept for interface stability

    dim3 block(32, 8);
    dim3 grid((w + 31) / 32, (h + 7) / 8);

    // d_seeds is repurposed as the per-pixel winner key buffer. 0xFF bytes =
    // 0xFFFFFFFF = kKeyEmpty, so every real stamp beats the initial value.
    CUDA_CHECK(cudaMemsetAsync(d_seeds, 0xFF, (size_t)w * h * sizeof(uint32_t), stream));

    unclip_scatter_kernel<<<grid, block, 0, stream>>>(
        d_seeds, d_bitmap, d_compact_ids, d_expand_per_comp, w, h);
    CUDA_CHECK(cudaGetLastError());

    unclip_resolve_kernel<<<grid, block, 0, stream>>>(
        d_expanded_labels, d_seeds, d_bitmap, d_compact_ids, d_expand_per_comp, w, h);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_accumulate_crack_perimeter(
    const int32_t *d_compact_ids, const uint8_t *d_bitmap,
    int w, int h, int num_slots, int *d_perim_per_comp, cudaStream_t stream) {
    if (num_slots <= 0) return;
    CUDA_CHECK(cudaMemsetAsync(d_perim_per_comp, 0,
                               (size_t)num_slots * sizeof(int), stream));
    dim3 block(32, 8);
    dim3 grid((w + 31) / 32, (h + 7) / 8);
    accumulate_crack_perimeter_kernel<<<grid, block, 0, stream>>>(
        d_compact_ids, d_bitmap, w, h, d_perim_per_comp);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_compute_expand_per_comp(
    const GpuDetBox *d_bboxes, const int *d_perim_per_comp, int num_slots,
    float unclip_ratio, float min_expand, float max_expand,
    float box_thresh, float *d_expand_per_comp, cudaStream_t stream) {
    if (num_slots <= 0) return;
    int block = 128;
    int grid = (num_slots + block - 1) / block;
    compute_expand_per_comp_kernel<<<grid, block, 0, stream>>>(
        d_bboxes, d_perim_per_comp, num_slots, unclip_ratio, min_expand, max_expand,
        box_thresh, d_expand_per_comp);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_jfa_extract_bboxes(const uint32_t *d_expanded_labels,
                             int w, int h,
                             GpuDetBox *d_bboxes, int num_slots,
                             cudaStream_t stream) {
    if (num_slots <= 0) return;
    int total = w * h;
    int block = 256;
    int init_grid = (num_slots + block - 1) / block;
    jfa_init_bboxes_kernel<<<init_grid, block, 0, stream>>>(d_bboxes, num_slots);
    CUDA_CHECK(cudaGetLastError());
    int grid = (total + block - 1) / block;
    jfa_extract_bboxes_kernel<<<grid, block, 0, stream>>>(
        d_expanded_labels, w, h, d_bboxes, num_slots);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace turbo_ocr::kernels
