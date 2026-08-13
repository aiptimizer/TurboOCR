// jfa.hip — HIP port of src/backends/nvidia/kernels_cuda/jfa.cu.
//
// Bounded per-component Euclidean unclip (DBNet polygon-offset), computed as a
// boundary-seeded disc dilation + atomicMin winner selection, plus the oriented
// (PCA) min-area-rect path. Bodies match the CUDA source bit-for-bit.
//
// AMD atomics used here:
//   * atomicMin(uint32_t*) on the packed (d2<<kLabelBits | cid) winner key — the
//     core of the unclip. Global 32-bit unsigned atomicMin is native on CDNA/RDNA.
//   * atomicAdd(unsigned long long*) for the PCA second moments — native 64-bit
//     integer atomics; supported, but 64-bit atomics are slower than 32-bit on
//     some gfx (see TODO).
//   * atomic_min_f/atomic_max_f: float min/max via atomicCAS on the int bit
//     pattern (__float_as_int/__int_as_float) — HIP provides these intrinsics.
//
// No warp/wavefront primitives and no cooperative launch here, so nothing depends
// on warpSize; these port cleanly. The only AMD note is 64-bit atomic throughput.

#include "kernels_hip.h"
#include "amd/support/hip_check.h"
#include <climits>
#include <cmath>
#include <hip/hip_runtime.h>

namespace turbo_ocr::amd::kernels {

static constexpr int kLabelBits = 13;
static constexpr uint32_t kLabelMask = (1u << kLabelBits) - 1u;
static constexpr uint32_t kKeyEmpty = 0xFFFFFFFFu;
static_assert(kMaxGpuComponents <= (int)kLabelMask, "label id must fit in key");

static constexpr int kMaxSafeExpand = 723;
static_assert(kMaxSafeExpand * kMaxSafeExpand < (1 << (32 - kLabelBits)),
              "seed radius must keep squared distance within the key budget");

__device__ __forceinline__ uint32_t pack_key(int d2, int cid) {
  return ((uint32_t)d2 << kLabelBits) | (uint32_t)cid;
}

__global__ void unclip_scatter_kernel(uint32_t *best, const uint8_t *bitmap,
                                      const int32_t *compact_ids,
                                      const float *expand_per_comp, int w,
                                      int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  int idx = y * w + x;
  if (!bitmap[idx]) return;

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
  float e = expand_per_comp[cid];
  if (e <= 0.0f) return;
  if (e > (float)kMaxSafeExpand) e = (float)kMaxSafeExpand;
  float e2 = e * e;
  int r = (int)ceilf(e);

  for (int dy = -r; dy <= r; ++dy) {
    int ty = y + dy;
    if (ty < 0 || ty >= h) continue;
    int dy2 = dy * dy;
    for (int dx = -r; dx <= r; ++dx) {
      int d2 = dx * dx + dy2;
      if ((float)d2 > e2) continue;
      int tx = x + dx;
      if (tx < 0 || tx >= w) continue;
      atomicMin(&best[ty * w + tx], pack_key(d2, cid));
    }
  }
}

__global__ void unclip_resolve_kernel(uint32_t *expanded_labels,
                                      const uint32_t *best, const uint8_t *bitmap,
                                      const int32_t *compact_ids,
                                      const float *expand_per_comp, int w, int h) {
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

__global__ void accumulate_crack_perimeter_kernel(const int32_t *compact_ids,
                                                  const uint8_t *bitmap, int w,
                                                  int h, int *perim_per_comp) {
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

__global__ void compute_expand_per_comp_kernel(
    const GpuDetBox *bboxes, const int *perim_per_comp, int num_slots,
    float unclip_ratio, float min_expand, float max_expand, float box_thresh,
    float *expand_per_comp) {
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

__global__ void jfa_init_bboxes_kernel(GpuDetBox *bboxes, int num_slots) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_slots) return;
  GpuDetBox &b = bboxes[i];
  b.xmin = INT_MAX; b.ymin = INT_MAX;
  b.xmax = INT_MIN; b.ymax = INT_MIN;
  b.pixel_count = 0;
  b.score = 0.0f;
}

__global__ void jfa_extract_bboxes_kernel(const uint32_t *expanded_labels, int w,
                                          int h, GpuDetBox *bboxes,
                                          int num_slots) {
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

__device__ __forceinline__ float atomic_min_f(float *addr, float val) {
  int *ai = (int *)addr;
  int old = *ai, assumed;
  do {
    assumed = old;
    old = atomicCAS(ai, assumed,
                    __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ __forceinline__ float atomic_max_f(float *addr, float val) {
  int *ai = (int *)addr;
  int old = *ai, assumed;
  do {
    assumed = old;
    old = atomicCAS(ai, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void oriented_extract_kernel(const uint32_t *expanded_labels, int w,
                                        int h, GpuDetBox *bboxes,
                                        unsigned long long *mom, int num_slots) {
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
  // TODO(on-hardware): 64-bit integer atomicAdd is native but slower than 32-bit
  // on some gfx; if the oriented path is hot, consider per-block LDS reduction of
  // the moments before the single global atomicAdd.
  const unsigned long long ux = (unsigned long long)x, uy = (unsigned long long)y;
  atomicAdd(&m[0], 1ULL);
  atomicAdd(&m[1], ux);
  atomicAdd(&m[2], uy);
  atomicAdd(&m[3], ux * ux);
  atomicAdd(&m[4], uy * uy);
  atomicAdd(&m[5], ux * uy);
}

__global__ void oriented_axis_kernel(const unsigned long long *mom,
                                     int num_slots, float *orient) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_slots) return;
  const unsigned long long *m = mom + (size_t)i * 6;
  float *o = orient + (size_t)i * 6;
  unsigned long long n = m[0];
  o[2] = INFINITY; o[3] = -INFINITY; o[4] = INFINITY; o[5] = -INFINITY;
  if (n < 3) { o[0] = 1.0f; o[1] = 0.0f; return; }
  double dn = (double)n;
  double mx = (double)m[1] / dn, my = (double)m[2] / dn;
  double cxx = (double)m[3] / dn - mx * mx;
  double cyy = (double)m[4] / dn - my * my;
  double cxy = (double)m[5] / dn - mx * my;
  double trace = cxx + cyy;
  double aniso = sqrt((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy);
  if (trace <= 0.0 || aniso < 0.05 * trace) { o[0] = 1.0f; o[1] = 0.0f; return; }
  double theta = 0.5 * atan2(2.0 * cxy, cxx - cyy);
  o[0] = (float)cos(theta);
  o[1] = (float)sin(theta);
}

__global__ void oriented_project_kernel(const uint32_t *expanded_labels, int w,
                                        int h, const unsigned long long *mom,
                                        float *orient, int num_slots) {
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
  const float uu[4] = {umin, umax, umax, umin};
  const float vv[4] = {vmin, vmin, vmax, vmax};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    b.ox[k] = uu[k] * c - vv[k] * s;
    b.oy[k] = uu[k] * s + vv[k] * c;
  }
}

void hip_jfa_extract_oriented(const uint32_t *d_expanded_labels, int w, int h,
                              GpuDetBox *d_bboxes, int num_slots,
                              unsigned long long *d_moments, float *d_orient,
                              hipStream_t stream) {
  if (num_slots <= 0) return;
  int total = w * h;
  int block = 256;
  int comp_grid = (num_slots + block - 1) / block;
  int pix_grid = (total + block - 1) / block;

  jfa_init_bboxes_kernel<<<comp_grid, block, 0, stream>>>(d_bboxes, num_slots);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemsetAsync(d_moments, 0,
                           (size_t)num_slots * 6 * sizeof(unsigned long long),
                           stream));

  oriented_extract_kernel<<<pix_grid, block, 0, stream>>>(
      d_expanded_labels, w, h, d_bboxes, d_moments, num_slots);
  HIP_CHECK(hipGetLastError());

  oriented_axis_kernel<<<comp_grid, block, 0, stream>>>(d_moments, num_slots, d_orient);
  HIP_CHECK(hipGetLastError());

  oriented_project_kernel<<<pix_grid, block, 0, stream>>>(
      d_expanded_labels, w, h, d_moments, d_orient, num_slots);
  HIP_CHECK(hipGetLastError());

  oriented_emit_kernel<<<comp_grid, block, 0, stream>>>(
      d_moments, d_orient, num_slots, d_bboxes);
  HIP_CHECK(hipGetLastError());
}

void hip_jfa_expand_labels(const uint8_t *d_bitmap, const int32_t *d_compact_ids,
                           const float *d_expand_per_comp,
                           uint32_t *d_expanded_labels, int w, int h,
                           float max_expand, uint32_t *d_seeds,
                           uint32_t *d_seeds_alt, hipStream_t stream) {
  (void)max_expand;
  (void)d_seeds_alt;

  dim3 block(32, 8);
  dim3 grid((w + 31) / 32, (h + 7) / 8);

  HIP_CHECK(hipMemsetAsync(d_seeds, 0xFF, (size_t)w * h * sizeof(uint32_t), stream));

  unclip_scatter_kernel<<<grid, block, 0, stream>>>(
      d_seeds, d_bitmap, d_compact_ids, d_expand_per_comp, w, h);
  HIP_CHECK(hipGetLastError());

  unclip_resolve_kernel<<<grid, block, 0, stream>>>(
      d_expanded_labels, d_seeds, d_bitmap, d_compact_ids, d_expand_per_comp, w, h);
  HIP_CHECK(hipGetLastError());
}

void hip_accumulate_crack_perimeter(const int32_t *d_compact_ids,
                                    const uint8_t *d_bitmap, int w, int h,
                                    int num_slots, int *d_perim_per_comp,
                                    hipStream_t stream) {
  if (num_slots <= 0) return;
  HIP_CHECK(hipMemsetAsync(d_perim_per_comp, 0, (size_t)num_slots * sizeof(int), stream));
  dim3 block(32, 8);
  dim3 grid((w + 31) / 32, (h + 7) / 8);
  accumulate_crack_perimeter_kernel<<<grid, block, 0, stream>>>(
      d_compact_ids, d_bitmap, w, h, d_perim_per_comp);
  HIP_CHECK(hipGetLastError());
}

void hip_compute_expand_per_comp(const GpuDetBox *d_bboxes,
                                 const int *d_perim_per_comp, int num_slots,
                                 float unclip_ratio, float min_expand,
                                 float max_expand, float box_thresh,
                                 float *d_expand_per_comp, hipStream_t stream) {
  if (num_slots <= 0) return;
  int block = 128;
  int grid = (num_slots + block - 1) / block;
  compute_expand_per_comp_kernel<<<grid, block, 0, stream>>>(
      d_bboxes, d_perim_per_comp, num_slots, unclip_ratio, min_expand,
      max_expand, box_thresh, d_expand_per_comp);
  HIP_CHECK(hipGetLastError());
}

void hip_jfa_extract_bboxes(const uint32_t *d_expanded_labels, int w, int h,
                            GpuDetBox *d_bboxes, int num_slots,
                            hipStream_t stream) {
  if (num_slots <= 0) return;
  int total = w * h;
  int block = 256;
  int init_grid = (num_slots + block - 1) / block;
  jfa_init_bboxes_kernel<<<init_grid, block, 0, stream>>>(d_bboxes, num_slots);
  HIP_CHECK(hipGetLastError());
  int grid = (total + block - 1) / block;
  jfa_extract_bboxes_kernel<<<grid, block, 0, stream>>>(
      d_expanded_labels, w, h, d_bboxes, num_slots);
  HIP_CHECK(hipGetLastError());
}

} // namespace turbo_ocr::amd::kernels
