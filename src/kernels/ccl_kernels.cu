#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/common/cuda/cuda_check.h"
#include "kernels_internal.cuh"
#include <climits>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace turbo_ocr::kernels {

// ==========================================================================
// GPU Connected Component Labeling (CCL) for text detection
// ==========================================================================
//
// Block-based Union-Find (BUF) — the Allegretti-Bolelli-Grana GPU-CCL state of
// the art, an evolution of the Komura-Aharoni equivalence and Playne-Russell
// union-find lineage. The image is tiled into independent 2x2 blocks: under
// 8-connectivity every foreground pixel inside a 2x2 block is mutually adjacent,
// so the whole block collapses to a single node in the union-find forest. That
// quarters the node count and the atomic/union traffic versus per-pixel CCL.
//
// Labeling is ONE lock-free hooking pass (no iterate-to-convergence, no
// whole-grid barrier loop) followed by ONE path-compression pass — the property
// that makes union-find CCL near-optimal against older iterated label-
// equivalence propagation. Each block examines only its 4 backward neighbours
// {NW, N, NE, W}; the forward duals are covered when those blocks run, so every
// block adjacency is unioned exactly once with no redundant passes.
//
// Pipeline:
//   1. init:     one node per 2x2 block, self-labelled if it holds any fg pixel
//   2. merge:    hook adjacent fg blocks via atomic union-find (single pass)
//   3. compress: path-compress each block node to its component root
//   4. compact:  dense component ids over root blocks (cooperative barrier),
//                then scatter to the per-pixel compact_ids map (fg only)
//   5. extract:  fused bbox + pred_map score accumulation
//   6. filter:   score/size threshold -> output boxes
//
// A block node is addressed by its top-left pixel's raster index, so find/union
// run on the shared int[w*h] label buffer exactly as a per-pixel forest would.
// The resulting component partition is identical to what a correct per-pixel
// 8-connected labeller produces — only the compact-id ordering differs, and no
// downstream consumer depends on that ordering (bbox = min/max, score = sum).
//
// MEMORY SAFETY:
// - ALL buffers pre-allocated by caller (no cudaMallocAsync)
// - d_labels / d_compact_ids are int[w*h] scratch; only fg pixels of
//   compact_ids end up >= 0 (bg = -1), as the JFA expand path requires
// - d_bboxes must be allocated for kMaxGpuComponents * 2 GpuDetBox
//   (first half for extraction, second half for filtered output)
// - float4 loads in threshold kernel require 16-byte alignment
//   (guaranteed by cudaMalloc which returns 256-byte aligned ptrs)
// - Only ONE cudaStreamSynchronize at the very end; no mid-pipeline host reads
// ==========================================================================

// --- Device helpers ---

__device__ __forceinline__ int ccl_find_root(const int *labels, int idx) {
  while (labels[idx] != idx)
    idx = labels[idx];
  return idx;
}

// Find with path-halving (ECL-CC's in-find compression): every hop rewrites
// idx's parent to its grandparent, so concurrent and subsequent traversals of
// the same chain get exponentially shorter. Lock-free safe: the store only
// ever replaces a parent with one of idx's current ANCESTORS (gp was reached
// by following parent pointers, and union-find only merges, never splits), so
// root-reachability is preserved under any interleaving; racing halvers may
// store different ancestors, all of them valid. Roots are never written here
// (p == idx returns first), so this cannot race the hooking CAS in ccl_union,
// which only targets labels[b] while labels[b] == b.
__device__ __forceinline__ int ccl_find_root_halve(int *labels, int idx) {
  while (true) {
    int p = labels[idx];
    if (p == idx) return idx;
    int gp = labels[p];
    if (p == gp) return p;
    labels[idx] = gp;  // path-halving
    idx = gp;
  }
}

// Lock-free union: hook the higher-index root onto the lower one, retrying the
// find on a CAS race. Standard concurrent union-find (Komura / ECL-CC style).
// Single-pass correctness: this call only returns after its two endpoints
// share a root — the loop exits either via a successful hook (CAS observed
// labels[b] == b and linked it) or via a == b (already connected), and a
// failed CAS retries with the freshly hooked target (b = find(old)). Since
// union-find connectivity is monotone (merges are never undone), one pass over
// every edge yields the transitive closure. This invariant is enforced
// empirically by the pathological-shape partition test in
// tests/cpp/test_gpu_safety.cpp (spirals, staircases, dense fuzz vs OpenCV).
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

__device__ __forceinline__ bool ccl_fg(const uint8_t *bitmap, int w,
                                        int x, int y) {
  return bitmap[y * w + x] != 0;
}

// --- Block-based Union-Find (BUF) kernels ---
// A 2x2 block at grid coord (bx, by) is addressed by the raster index of its
// top-left pixel: (2*by)*w + 2*bx. That top-left pixel is always in bounds
// because nbx = ceil(w/2), nby = ceil(h/2).

// Step 1: init one union-find node per 2x2 block (self if any fg pixel, else bg)
__global__ __launch_bounds__(256)
void ccl_buf_init_kernel(const uint8_t * __restrict__ bitmap,
                         int * __restrict__ labels,
                         int w, int h, int nbx, int nby) {
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

// Step 2: single hooking pass. Union this fg block with its backward fg
// neighbours over the exact 8-connected block adjacencies. Pixels of a block:
//   P0 P1     at (x0,y0) (x0+1,y0)
//   P2 P3        (x0,y0+1) (x0+1,y0+1)
__global__ __launch_bounds__(256)
void ccl_buf_merge_kernel(const uint8_t * __restrict__ bitmap,
                          int * __restrict__ labels,
                          int w, int h, int nbx, int nby) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= nbx || by >= nby) return;
  int x0 = 2 * bx, y0 = 2 * by;
  int rep = y0 * w + x0;
  if (labels[rep] < 0) return;  // background block

  bool p0 = ccl_fg(bitmap, w, x0, y0);
  bool p1 = (x0 + 1 < w) && ccl_fg(bitmap, w, x0 + 1, y0);
  bool p2 = (y0 + 1 < h) && ccl_fg(bitmap, w, x0, y0 + 1);

  if (by > 0) {
    // N block: X top row (P0,P1) vs N bottom row — all four pairs 8-adjacent.
    bool n2 = ccl_fg(bitmap, w, x0, y0 - 1);
    bool n3 = (x0 + 1 < w) && ccl_fg(bitmap, w, x0 + 1, y0 - 1);
    if ((p0 || p1) && (n2 || n3))
      ccl_union(labels, rep, (y0 - 2) * w + x0);
    // NW block: only the X.P0 ~ NW.P3 corner touches.
    if (bx > 0 && p0 && ccl_fg(bitmap, w, x0 - 1, y0 - 1))
      ccl_union(labels, rep, (y0 - 2) * w + (x0 - 2));
    // NE block: only the X.P1 ~ NE.P2 corner touches.
    if (bx + 1 < nbx && p1 && (x0 + 2 < w) && ccl_fg(bitmap, w, x0 + 2, y0 - 1))
      ccl_union(labels, rep, (y0 - 2) * w + (x0 + 2));
  }
  if (bx > 0) {
    // W block: X left col (P0,P2) vs W right col — all four pairs 8-adjacent.
    bool w1 = ccl_fg(bitmap, w, x0 - 1, y0);
    bool w3 = (y0 + 1 < h) && ccl_fg(bitmap, w, x0 - 1, y0 + 1);
    if ((p0 || p2) && (w1 || w3))
      ccl_union(labels, rep, y0 * w + (x0 - 2));
  }
}

// Step 3: path-compress each fg block node to its component root.
__global__ __launch_bounds__(256)
void ccl_buf_compress_kernel(int * __restrict__ labels,
                             int w, int nbx, int nby) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= nbx || by >= nby) return;
  int rep = (2 * by) * w + (2 * bx);
  if (labels[rep] < 0) return;
  labels[rep] = ccl_find_root(labels, rep);
}

// Step 4a: dense component ids over root blocks, then propagate to non-root
// blocks. Cooperative grid.sync() keeps the two phases device-resident.
// Grid is limited to the cooperative maximum; a stride loop covers all blocks.
__global__ __launch_bounds__(256)
void ccl_buf_compact_assign_kernel(const int * __restrict__ labels,
                                   int * __restrict__ compact_ids,
                                   int * __restrict__ id_counter,
                                   int w, int nbx, int nby,
                                   int max_components) {
  auto grid = cg::this_grid();
  int nblocks = nbx * nby;
  int stride = gridDim.x * blockDim.x;

  for (int bi = blockIdx.x * blockDim.x + threadIdx.x; bi < nblocks;
       bi += stride) {
    int rep = (2 * (bi / nbx)) * w + 2 * (bi % nbx);
    if (labels[rep] == rep) {  // root of a foreground component
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

// Step 4b: expand block-level ids to the per-pixel compact_ids map. Foreground
// pixels get their block's component id; every other pixel (incl. bg pixels
// inside a fg block) gets -1, so extract counts only true foreground.
__global__ __launch_bounds__(256)
void ccl_buf_scatter_kernel(const uint8_t * __restrict__ bitmap,
                            const int * __restrict__ labels,
                            int * __restrict__ compact_ids,
                            int w, int h, int nbx, int nby) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= nbx || by >= nby) return;
  int x0 = 2 * bx, y0 = 2 * by;
  int rep = y0 * w + x0;
  int cid = (labels[rep] >= 0) ? compact_ids[rep] : -1;  // read before overwrite
  compact_ids[rep] = ccl_fg(bitmap, w, x0, y0) ? cid : -1;
  if (x0 + 1 < w)
    compact_ids[rep + 1] = ccl_fg(bitmap, w, x0 + 1, y0) ? cid : -1;
  if (y0 + 1 < h)
    compact_ids[rep + w] = ccl_fg(bitmap, w, x0, y0 + 1) ? cid : -1;
  if (x0 + 1 < w && y0 + 1 < h)
    compact_ids[rep + w + 1] = ccl_fg(bitmap, w, x0 + 1, y0 + 1) ? cid : -1;
}

// Step 5: Init bboxes on GPU (kernel, not host memcpy -- no sync needed)
__global__ __launch_bounds__(256)
void ccl_init_bboxes_kernel(GpuDetBox * __restrict__ bboxes, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  bboxes[idx].xmin = INT_MAX;
  bboxes[idx].ymin = INT_MAX;
  bboxes[idx].xmax = 0;
  bboxes[idx].ymax = 0;
  bboxes[idx].pixel_count = 0;
  bboxes[idx].score = 0.0f;
}

// Step 6: Fused extract bboxes + accumulate score sum (single memory pass)
// score field accumulates raw pred_map sum; divided by pixel_count in filter.
//
// OPTIMIZED: shared-memory per-block accumulation with a fixed-size component
// hash table (32 slots). Each block accumulates local bbox/score in shared
// memory, then ONE thread per slot flushes to global with atomics.
// Reduces global atomics from N_pixels * 6 to N_blocks * 6 per component.
//
// Fallback: if a block sees >32 unique components (rare), excess pixels
// fall back to direct global atomics.

static constexpr int kExtractSlots = 32;       // shared-memory hash table size
static constexpr int kExtractSlotsMask = 31;    // kExtractSlots - 1

struct ExtractSlot {
  int cid;          // component ID (-1 = empty)
  int xmin, xmax;
  int ymin, ymax;
  int pixel_count;
  float score_sum;
};

__global__ __launch_bounds__(256)
void ccl_fused_extract_kernel(const int * __restrict__ compact_ids,
                               const float * __restrict__ pred_map,
                               GpuDetBox * __restrict__ bboxes,
                               int w, int h, int total) {
  __shared__ ExtractSlot slots[kExtractSlots];

  int tid = threadIdx.x;

  // Initialize shared memory slots
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

      // Open-addressing hash: linear probe to find or claim a slot
      int hash = (unsigned int)cid & kExtractSlotsMask;
      bool inserted = false;
      for (int probe = 0; probe < kExtractSlots; probe++) {
        int slot_idx = (hash + probe) & kExtractSlotsMask;

        // Try to claim an empty slot with atomicCAS
        int old = atomicCAS(&slots[slot_idx].cid, -1, cid);
        if (old == -1 || old == cid) {
          // We own this slot (either claimed it or it was already ours)
          atomicMin(&slots[slot_idx].xmin, x);
          atomicMax(&slots[slot_idx].xmax, x);
          atomicMin(&slots[slot_idx].ymin, y);
          atomicMax(&slots[slot_idx].ymax, y);
          atomicAdd(&slots[slot_idx].pixel_count, 1);
          atomicAdd(&slots[slot_idx].score_sum, score_val);
          inserted = true;
          break;
        }
        // Slot taken by different cid, continue probing
      }

      // Fallback: hash table full (>32 unique components in this block)
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

  // Flush shared memory slots to global memory: one thread per slot
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

// Step 7: Filter (launched with kMaxGpuComponents threads -- no CPU sync needed)
// Unused slots have pixel_count=0 from init, so they early-return.
__global__ __launch_bounds__(256)
void ccl_filter_kernel(const GpuDetBox * __restrict__ bboxes,
                        float box_thresh,
                        GpuDetBox * __restrict__ out_bboxes,
                        int * __restrict__ out_count,
                        int max_components, int max_out) {
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

// ==========================================================================
// Host wrapper: full GPU CCL pipeline
// ALL buffers pre-allocated. No cudaMallocAsync/cudaFreeAsync.
// ONE cudaStreamSynchronize at the end.
// ==========================================================================

int cuda_gpu_ccl_detect(
    const uint8_t *d_bitmap,
    const float *d_pred_map,
    int w, int h,
    float box_thresh,
    int *d_labels,
    int *d_compact_ids,
    int *d_id_counter,
    GpuDetBox *d_bboxes,
    int *d_num_boxes,
    GpuDetBox *h_boxes,
    int *h_num_boxes,
    cudaStream_t stream,
    int *h_num_total) {

  int total = w * h;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  // 2x2-block grid for the BUF core (nbx * nby = ceil(w/2) * ceil(h/2) nodes).
  int nbx = (w + 1) / 2;
  int nby = (h + 1) / 2;
  dim3 bblock(32, 8);
  dim3 bgrid((nbx + bblock.x - 1) / bblock.x,
             (nby + bblock.y - 1) / bblock.y);

  // Step 1: init one union-find node per 2x2 block.
  ccl_buf_init_kernel<<<bgrid, bblock, 0, stream>>>(
      d_bitmap, d_labels, w, h, nbx, nby);
  CUDA_CHECK(cudaGetLastError());

  // Step 2: single lock-free hooking pass over the 8-connected block
  // adjacencies. No iterate-to-convergence, no whole-grid barrier loop.
  ccl_buf_merge_kernel<<<bgrid, bblock, 0, stream>>>(
      d_bitmap, d_labels, w, h, nbx, nby);
  CUDA_CHECK(cudaGetLastError());

  // Step 3: path-compress block nodes to their component roots.
  ccl_buf_compress_kernel<<<bgrid, bblock, 0, stream>>>(d_labels, w, nbx, nby);
  CUDA_CHECK(cudaGetLastError());

  // Step 4: dense component ids over root blocks (cooperative grid.sync between
  // assign and propagate), then scatter block ids to the per-pixel map.
  CUDA_CHECK(cudaMemsetAsync(d_id_counter, 0, sizeof(int), stream));
  {
    int nblocks = nbx * nby;
    int need = (nblocks + threads - 1) / threads;
    int coop_grid = coop_grid_for(ccl_buf_compact_assign_kernel, threads);
    if (coop_grid > need) coop_grid = need;
    if (coop_grid < 1) coop_grid = 1;

    int max_comp = kMaxGpuComponents;
    void *args[] = { (void*)&d_labels, (void*)&d_compact_ids,
                     (void*)&d_id_counter, (void*)&w, (void*)&nbx,
                     (void*)&nby, (void*)&max_comp };
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)ccl_buf_compact_assign_kernel, dim3(coop_grid), dim3(threads),
        args, 0, stream));
    CUDA_CHECK(cudaGetLastError());
  }
  ccl_buf_scatter_kernel<<<bgrid, bblock, 0, stream>>>(
      d_bitmap, d_labels, d_compact_ids, w, h, nbx, nby);
  CUDA_CHECK(cudaGetLastError());

  // Step 5: Init bboxes via GPU kernel (not host memcpy -- no contention)
  {
    int bbox_blocks = (kMaxGpuComponents + threads - 1) / threads;
    ccl_init_bboxes_kernel<<<bbox_blocks, threads, 0, stream>>>(
        d_bboxes, kMaxGpuComponents);
    CUDA_CHECK(cudaGetLastError());
  }

  // Step 6: Fused extract bboxes + accumulate score
  ccl_fused_extract_kernel<<<blocks, threads, 0, stream>>>(
      d_compact_ids, d_pred_map, d_bboxes, w, h, total);
  CUDA_CHECK(cudaGetLastError());

  // Step 7: Filter (launch kMaxGpuComponents threads -- no CPU read needed)
  CUDA_CHECK(cudaMemsetAsync(d_num_boxes, 0, sizeof(int), stream));
  GpuDetBox *d_out_bboxes = d_bboxes + kMaxGpuComponents;
  {
    int filter_blocks = (kMaxGpuComponents + threads - 1) / threads;
    ccl_filter_kernel<<<filter_blocks, threads, 0, stream>>>(
        d_bboxes, box_thresh, d_out_bboxes, d_num_boxes,
        kMaxGpuComponents, kMaxGpuComponents);
    CUDA_CHECK(cudaGetLastError());
  }

  // === SINGLE SYNC: copy count + boxes in one batch, one sync ===
  CUDA_CHECK(cudaMemcpyAsync(h_num_boxes, d_num_boxes, sizeof(int),
                              cudaMemcpyDeviceToHost, stream));
  // Optional: pre-filter component total for callers that index by pre-filter
  // compact_id (e.g. the JFA per-component expand path).
  if (h_num_total != nullptr) {
    CUDA_CHECK(cudaMemcpyAsync(h_num_total, d_id_counter, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
  }
  CUDA_CHECK(cudaMemcpyAsync(h_boxes, d_out_bboxes,
                              kMaxGpuComponents * sizeof(GpuDetBox),
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int h_count = std::min(*h_num_boxes, (int)kMaxGpuComponents);

  *h_num_boxes = h_count;
  return h_count;
}

} // namespace turbo_ocr::kernels
