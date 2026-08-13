// ccl_kernels.hip — HIP port of src/backends/nvidia/kernels_cuda/ccl_kernels.cu.
//
// Block-based Union-Find (BUF) GPU connected-component labelling for DB text
// detection. Kernel bodies match the CUDA source bit-for-bit; the atomics
// (atomicCAS/atomicMin/atomicMax/atomicAdd on global AND shared int/float),
// the cooperative grid.sync compaction, and the open-addressing shared hash in
// the fused-extract kernel all port to HIP — but two AMD caveats need on-hardware
// attention. See the two TODO blocks below.

#include "kernels_hip.h"
#include "kernels_internal_hip.h"
#include "amd/support/hip_check.h"
#include <algorithm>
#include <climits>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

namespace turbo_ocr::amd::kernels {

// ============================================================================
// AMD CAVEAT 1 — cooperative launch (ccl_buf_compact_assign_kernel). HANDLED.
// ----------------------------------------------------------------------------
// grid.sync() requires a COOPERATIVE launch (hipLaunchCooperativeKernel) AND a
// gfx target + driver that supports it. Not all ROCm devices report
// hipDeviceAttributeCooperativeLaunch == 1 — notably several consumer RDNA parts
// and older ROCm stacks. A port that launches cooperatively unconditionally
// simply FAILS TO DETECT TEXT on those cards (the launch errors out, the compact
// ids are never assigned, every downstream component id is garbage).
//
// FIX APPLIED: the compaction now has two interchangeable implementations and
// the host wrapper picks one from a CACHED device-attribute probe:
//   * coop == 1 -> the single cooperative kernel with grid.sync() (one launch).
//   * coop == 0 -> ccl_buf_compact_assign_pass1 then ..._pass2, TWO ordINARY
//     launches on the SAME stream. Two kernels on one stream are implicitly
//     ordered by the HIP execution model, and pass2 only reads what pass1 wrote,
//     so the grid-wide barrier is expressed by the launch boundary itself — no
//     hipStreamSynchronize, no host round-trip, no extra cost beyond one launch.
// The arithmetic in both passes is copied verbatim from the cooperative kernel;
// the ONLY difference is where the barrier comes from. Keep them in sync.
//
// AMD CAVEAT 2 — shared-memory atomics (ccl_fused_extract_kernel).
// ----------------------------------------------------------------------------
// The fused-extract kernel does atomicCAS/atomicMin/atomicMax/atomicAdd into a
// __shared__ ExtractSlot[32] hash table. HIP supports LDS (shared) atomics on
// CDNA and RDNA, but atomicMin/atomicMax on shared FLOAT is emulated via CAS on
// some gfx and shared-atomic throughput differs markedly from NVIDIA. The
// algorithm is correct as written; performance may need retuning (slot count,
// or moving the per-block reduction to a wavefront-shuffle scheme).
// TODO(on-hardware): profile shared-atomic contention; if it dominates, replace
// the 32-slot LDS hash with a wavefront-ballot reduction keyed on cid.
// ============================================================================

__device__ __forceinline__ int ccl_find_root(const int *labels, int idx) {
  while (labels[idx] != idx)
    idx = labels[idx];
  return idx;
}

__device__ __forceinline__ int ccl_find_root_halve(int *labels, int idx) {
  while (true) {
    int p = labels[idx];
    if (p == idx) return idx;
    int gp = labels[p];
    if (p == gp) return p;
    labels[idx] = gp; // path-halving
    idx = gp;
  }
}

__device__ __forceinline__ void ccl_union(int *labels, int a, int b) {
  a = ccl_find_root_halve(labels, a);
  b = ccl_find_root_halve(labels, b);
  while (a != b) {
    if (a > b) { int t = a; a = b; b = t; }
    int old = atomicCAS(&labels[b], b, a);
    if (old == b) break;
    a = ccl_find_root_halve(labels, a);
    b = ccl_find_root_halve(labels, old);
  }
}

__device__ __forceinline__ bool ccl_fg(const uint8_t *bitmap, int w, int x,
                                        int y) {
  return bitmap[y * w + x] != 0;
}

__global__ __launch_bounds__(256)
void ccl_buf_init_kernel(const uint8_t *__restrict__ bitmap,
                         int *__restrict__ labels, int w, int h, int nbx,
                         int nby) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= nbx || by >= nby) return;
  int x0 = 2 * bx, y0 = 2 * by;
  bool has_fg = ccl_fg(bitmap, w, x0, y0)
      || ((x0 + 1 < w) && ccl_fg(bitmap, w, x0 + 1, y0))
      || ((y0 + 1 < h) && ccl_fg(bitmap, w, x0, y0 + 1))
      || ((x0 + 1 < w) && (y0 + 1 < h) && ccl_fg(bitmap, w, x0 + 1, y0 + 1));
  int rep = y0 * w + x0;
  labels[rep] = has_fg ? rep : -1;
}

__global__ __launch_bounds__(256)
void ccl_buf_merge_kernel(const uint8_t *__restrict__ bitmap,
                          int *__restrict__ labels, int w, int h, int nbx,
                          int nby) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= nbx || by >= nby) return;
  int x0 = 2 * bx, y0 = 2 * by;
  int rep = y0 * w + x0;
  if (labels[rep] < 0) return;

  bool p0 = ccl_fg(bitmap, w, x0, y0);
  bool p1 = (x0 + 1 < w) && ccl_fg(bitmap, w, x0 + 1, y0);
  bool p2 = (y0 + 1 < h) && ccl_fg(bitmap, w, x0, y0 + 1);

  if (by > 0) {
    bool n2 = ccl_fg(bitmap, w, x0, y0 - 1);
    bool n3 = (x0 + 1 < w) && ccl_fg(bitmap, w, x0 + 1, y0 - 1);
    if ((p0 || p1) && (n2 || n3))
      ccl_union(labels, rep, (y0 - 2) * w + x0);
    if (bx > 0 && p0 && ccl_fg(bitmap, w, x0 - 1, y0 - 1))
      ccl_union(labels, rep, (y0 - 2) * w + (x0 - 2));
    if (bx + 1 < nbx && p1 && (x0 + 2 < w) && ccl_fg(bitmap, w, x0 + 2, y0 - 1))
      ccl_union(labels, rep, (y0 - 2) * w + (x0 + 2));
  }
  if (bx > 0) {
    bool w1 = ccl_fg(bitmap, w, x0 - 1, y0);
    bool w3 = (y0 + 1 < h) && ccl_fg(bitmap, w, x0 - 1, y0 + 1);
    if ((p0 || p2) && (w1 || w3))
      ccl_union(labels, rep, y0 * w + (x0 - 2));
  }
}

__global__ __launch_bounds__(256)
void ccl_buf_compress_kernel(int *__restrict__ labels, int w, int nbx, int nby) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= nbx || by >= nby) return;
  int rep = (2 * by) * w + (2 * bx);
  if (labels[rep] < 0) return;
  labels[rep] = ccl_find_root(labels, rep);
}

__global__ __launch_bounds__(256)
void ccl_buf_compact_assign_kernel(const int *__restrict__ labels,
                                   int *__restrict__ compact_ids,
                                   int *__restrict__ id_counter, int w, int nbx,
                                   int nby, int max_components) {
  auto grid = cg::this_grid();
  int nblocks = nbx * nby;
  int stride = gridDim.x * blockDim.x;

  for (int bi = blockIdx.x * blockDim.x + threadIdx.x; bi < nblocks;
       bi += stride) {
    int rep = (2 * (bi / nbx)) * w + 2 * (bi % nbx);
    if (labels[rep] == rep) {
      int cid = atomicAdd(id_counter, 1);
      compact_ids[rep] = (cid < max_components) ? cid : -1;
    }
  }
  grid.sync();
  for (int bi = blockIdx.x * blockDim.x + threadIdx.x; bi < nblocks;
       bi += stride) {
    int rep = (2 * (bi / nbx)) * w + 2 * (bi % nbx);
    int root = labels[rep];
    if (root >= 0 && root != rep)
      compact_ids[rep] = compact_ids[root];
  }
}

// --- Non-cooperative equivalent (AMD CAVEAT 1 fallback) ---------------------
// pass1 == the pre-grid.sync() half, pass2 == the post-grid.sync() half, split
// at the barrier. Launched back-to-back on one stream; the launch boundary IS
// the grid-wide barrier. Bodies are verbatim copies — if you edit the
// cooperative kernel above, edit these too.

__global__ __launch_bounds__(256)
void ccl_buf_compact_assign_pass1(const int *__restrict__ labels,
                                  int *__restrict__ compact_ids,
                                  int *__restrict__ id_counter, int w, int nbx,
                                  int nby, int max_components) {
  int nblocks = nbx * nby;
  int stride = gridDim.x * blockDim.x;
  for (int bi = blockIdx.x * blockDim.x + threadIdx.x; bi < nblocks;
       bi += stride) {
    int rep = (2 * (bi / nbx)) * w + 2 * (bi % nbx);
    if (labels[rep] == rep) {
      int cid = atomicAdd(id_counter, 1);
      compact_ids[rep] = (cid < max_components) ? cid : -1;
    }
  }
}

__global__ __launch_bounds__(256)
void ccl_buf_compact_assign_pass2(const int *__restrict__ labels,
                                  int *__restrict__ compact_ids, int w, int nbx,
                                  int nby) {
  int nblocks = nbx * nby;
  int stride = gridDim.x * blockDim.x;
  for (int bi = blockIdx.x * blockDim.x + threadIdx.x; bi < nblocks;
       bi += stride) {
    int rep = (2 * (bi / nbx)) * w + 2 * (bi % nbx);
    int root = labels[rep];
    if (root >= 0 && root != rep)
      compact_ids[rep] = compact_ids[root];
  }
}

namespace {
// Cached per-device cooperative-launch support probe. Queried once; the answer
// is a static property of the (device, driver) pair, so this never runs on the
// hot path after the first detection.
bool device_supports_cooperative_launch() {
  static thread_local int cached_dev = -1;
  static thread_local bool cached_ok = false;
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess)
    return false;
  if (dev != cached_dev) {
    int coop = 0;
    cached_ok = (hipDeviceGetAttribute(&coop,
                                       hipDeviceAttributeCooperativeLaunch,
                                       dev) == hipSuccess) &&
                coop != 0;
    cached_dev = dev;
  }
  return cached_ok;
}
} // namespace

__global__ __launch_bounds__(256)
void ccl_buf_scatter_kernel(const uint8_t *__restrict__ bitmap,
                            const int *__restrict__ labels,
                            int *__restrict__ compact_ids, int w, int h, int nbx,
                            int nby) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= nbx || by >= nby) return;
  int x0 = 2 * bx, y0 = 2 * by;
  int rep = y0 * w + x0;
  int cid = (labels[rep] >= 0) ? compact_ids[rep] : -1;
  compact_ids[rep] = ccl_fg(bitmap, w, x0, y0) ? cid : -1;
  if (x0 + 1 < w)
    compact_ids[rep + 1] = ccl_fg(bitmap, w, x0 + 1, y0) ? cid : -1;
  if (y0 + 1 < h)
    compact_ids[rep + w] = ccl_fg(bitmap, w, x0, y0 + 1) ? cid : -1;
  if (x0 + 1 < w && y0 + 1 < h)
    compact_ids[rep + w + 1] = ccl_fg(bitmap, w, x0 + 1, y0 + 1) ? cid : -1;
}

__global__ __launch_bounds__(256)
void ccl_init_bboxes_kernel(GpuDetBox *__restrict__ bboxes, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  bboxes[idx].xmin = INT_MAX;
  bboxes[idx].ymin = INT_MAX;
  bboxes[idx].xmax = 0;
  bboxes[idx].ymax = 0;
  bboxes[idx].pixel_count = 0;
  bboxes[idx].score = 0.0f;
}

static constexpr int kExtractSlots = 32;
static constexpr int kExtractSlotsMask = 31;

struct ExtractSlot {
  int cid;
  int xmin, xmax;
  int ymin, ymax;
  int pixel_count;
  float score_sum;
};

__global__ __launch_bounds__(256)
void ccl_fused_extract_kernel(const int *__restrict__ compact_ids,
                              const float *__restrict__ pred_map,
                              GpuDetBox *__restrict__ bboxes, int w, int h,
                              int total) {
  __shared__ ExtractSlot slots[kExtractSlots];
  int tid = threadIdx.x;

  if (tid < kExtractSlots) {
    slots[tid].cid = -1;
    slots[tid].xmin = INT_MAX;
    slots[tid].xmax = 0;
    slots[tid].ymin = INT_MAX;
    slots[tid].ymax = 0;
    slots[tid].pixel_count = 0;
    slots[tid].score_sum = 0.0f;
  }
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    int cid = compact_ids[idx];
    if (cid >= 0) {
      int x = idx % w;
      int y = idx / w;
      float score_val = pred_map[idx];

      int hash = (unsigned int)cid & kExtractSlotsMask;
      bool inserted = false;
      for (int probe = 0; probe < kExtractSlots; probe++) {
        int slot_idx = (hash + probe) & kExtractSlotsMask;
        int old = atomicCAS(&slots[slot_idx].cid, -1, cid);
        if (old == -1 || old == cid) {
          atomicMin(&slots[slot_idx].xmin, x);
          atomicMax(&slots[slot_idx].xmax, x);
          atomicMin(&slots[slot_idx].ymin, y);
          atomicMax(&slots[slot_idx].ymax, y);
          atomicAdd(&slots[slot_idx].pixel_count, 1);
          atomicAdd(&slots[slot_idx].score_sum, score_val);
          inserted = true;
          break;
        }
      }

      if (!inserted) {
        atomicMin(&bboxes[cid].xmin, x);
        atomicMax(&bboxes[cid].xmax, x);
        atomicMin(&bboxes[cid].ymin, y);
        atomicMax(&bboxes[cid].ymax, y);
        atomicAdd(&bboxes[cid].pixel_count, 1);
        atomicAdd(&bboxes[cid].score, score_val);
      }
    }
  }
  __syncthreads();

  if (tid < kExtractSlots) {
    int cid = slots[tid].cid;
    if (cid >= 0 && slots[tid].pixel_count > 0) {
      atomicMin(&bboxes[cid].xmin, slots[tid].xmin);
      atomicMax(&bboxes[cid].xmax, slots[tid].xmax);
      atomicMin(&bboxes[cid].ymin, slots[tid].ymin);
      atomicMax(&bboxes[cid].ymax, slots[tid].ymax);
      atomicAdd(&bboxes[cid].pixel_count, slots[tid].pixel_count);
      atomicAdd(&bboxes[cid].score, slots[tid].score_sum);
    }
  }
}

__global__ __launch_bounds__(256)
void ccl_filter_kernel(const GpuDetBox *__restrict__ bboxes, float box_thresh,
                       GpuDetBox *__restrict__ out_bboxes,
                       int *__restrict__ out_count, int max_components,
                       int max_out) {
  int cid = blockIdx.x * blockDim.x + threadIdx.x;
  if (cid >= max_components) return;

  GpuDetBox box = bboxes[cid];
  if (box.pixel_count < 3) return;

  int bw = box.xmax - box.xmin + 1;
  int bh = box.ymax - box.ymin + 1;
  if (bw < 3 || bh < 3) return;

  float score = box.score / (float)box.pixel_count;
  if (score < box_thresh) return;

  box.score = score;
  int out_idx = atomicAdd(out_count, 1);
  if (out_idx < max_out)
    out_bboxes[out_idx] = box;
}

int hip_gpu_ccl_detect(const uint8_t *d_bitmap, const float *d_pred_map, int w,
                       int h, float box_thresh, int *d_labels,
                       int *d_compact_ids, int *d_id_counter,
                       GpuDetBox *d_bboxes, int *d_num_boxes, GpuDetBox *h_boxes,
                       int *h_num_boxes, hipStream_t stream, int *h_num_total) {
  int total = w * h;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  int nbx = (w + 1) / 2;
  int nby = (h + 1) / 2;
  dim3 bblock(32, 8);
  dim3 bgrid((nbx + bblock.x - 1) / bblock.x, (nby + bblock.y - 1) / bblock.y);

  ccl_buf_init_kernel<<<bgrid, bblock, 0, stream>>>(d_bitmap, d_labels, w, h, nbx, nby);
  HIP_CHECK(hipGetLastError());

  ccl_buf_merge_kernel<<<bgrid, bblock, 0, stream>>>(d_bitmap, d_labels, w, h, nbx, nby);
  HIP_CHECK(hipGetLastError());

  ccl_buf_compress_kernel<<<bgrid, bblock, 0, stream>>>(d_labels, w, nbx, nby);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemsetAsync(d_id_counter, 0, sizeof(int), stream));
  {
    int nblocks = nbx * nby;
    int need = (nblocks + threads - 1) / threads;
    int coop_grid = coop_grid_for(ccl_buf_compact_assign_kernel, threads);
    if (coop_grid > need) coop_grid = need;
    if (coop_grid < 1) coop_grid = 1;

    int max_comp = kMaxGpuComponents;
    if (device_supports_cooperative_launch()) {
      void *args[] = {(void *)&d_labels, (void *)&d_compact_ids,
                      (void *)&d_id_counter, (void *)&w, (void *)&nbx,
                      (void *)&nby, (void *)&max_comp};
      HIP_CHECK(hipLaunchCooperativeKernel(
          (void *)ccl_buf_compact_assign_kernel, dim3(coop_grid), dim3(threads),
          args, 0, stream));
      HIP_CHECK(hipGetLastError());
    } else {
      // AMD CAVEAT 1 fallback: the launch boundary between pass1 and pass2 on
      // this single stream provides the grid-wide barrier that grid.sync()
      // would have. `need` (not coop_grid) is used because a non-cooperative
      // launch has no occupancy ceiling to respect.
      ccl_buf_compact_assign_pass1<<<need, threads, 0, stream>>>(
          d_labels, d_compact_ids, d_id_counter, w, nbx, nby, max_comp);
      HIP_CHECK(hipGetLastError());
      ccl_buf_compact_assign_pass2<<<need, threads, 0, stream>>>(
          d_labels, d_compact_ids, w, nbx, nby);
      HIP_CHECK(hipGetLastError());
    }
  }
  ccl_buf_scatter_kernel<<<bgrid, bblock, 0, stream>>>(d_bitmap, d_labels, d_compact_ids, w, h, nbx, nby);
  HIP_CHECK(hipGetLastError());

  {
    int bbox_blocks = (kMaxGpuComponents + threads - 1) / threads;
    ccl_init_bboxes_kernel<<<bbox_blocks, threads, 0, stream>>>(d_bboxes, kMaxGpuComponents);
    HIP_CHECK(hipGetLastError());
  }

  ccl_fused_extract_kernel<<<blocks, threads, 0, stream>>>(d_compact_ids, d_pred_map, d_bboxes, w, h, total);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemsetAsync(d_num_boxes, 0, sizeof(int), stream));
  GpuDetBox *d_out_bboxes = d_bboxes + kMaxGpuComponents;
  {
    int filter_blocks = (kMaxGpuComponents + threads - 1) / threads;
    ccl_filter_kernel<<<filter_blocks, threads, 0, stream>>>(
        d_bboxes, box_thresh, d_out_bboxes, d_num_boxes, kMaxGpuComponents,
        kMaxGpuComponents);
    HIP_CHECK(hipGetLastError());
  }

  HIP_CHECK(hipMemcpyAsync(h_num_boxes, d_num_boxes, sizeof(int),
                           hipMemcpyDeviceToHost, stream));
  if (h_num_total != nullptr) {
    HIP_CHECK(hipMemcpyAsync(h_num_total, d_id_counter, sizeof(int),
                             hipMemcpyDeviceToHost, stream));
  }
  HIP_CHECK(hipMemcpyAsync(h_boxes, d_out_bboxes,
                           kMaxGpuComponents * sizeof(GpuDetBox),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int h_count = std::min(*h_num_boxes, (int)kMaxGpuComponents);
  *h_num_boxes = h_count;
  return h_count;
}

} // namespace turbo_ocr::amd::kernels
