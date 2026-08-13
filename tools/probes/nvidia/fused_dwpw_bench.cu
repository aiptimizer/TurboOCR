// Microbenchmark: fused depthwise(3x3)->pointwise(1x1) kernel vs cuDNN's two
// separate convs, FP16, NHWC, on sm_120. Answers the gating question for the
// fused-DWPW TensorRT-plugin idea: does keeping the depthwise intermediate
// on-chip (no DRAM round-trip) actually beat the cuDNN layer-by-layer baseline
// on Blackwell's high memory bandwidth, at real PP-LCNetV3 block shapes?
//
// Build: nvcc -O3 -arch=sm_120 tools/probes/nvidia/fused_dwpw_bench.cu -o build/fused_dwpw_bench \
//          -lcudnn -lcudart
// Run:   build/fused_dwpw_bench N C Cout H W   (defaults to a det_tiny-like block)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CUDNN_CHECK(x) do{ cudnnStatus_t s=(x); if(s!=CUDNN_STATUS_SUCCESS){ \
  printf("cuDNN error %s:%d: %s\n",__FILE__,__LINE__,cudnnGetErrorString(s)); exit(1);} }while(0)

// ---- fused kernel ----------------------------------------------------------
// Layout NHWC. dwW[C][3][3] (depthwise), pwW[Cout][C] (pointwise 1x1).
// One block = TILE_H x TILE_W output tile, one image n. Each thread owns one
// output pixel and accumulates all Cout channels in shared memory while the
// input is streamed in CHUNK-channel slabs through shared memory. The depthwise
// result is consumed immediately into the pointwise accumulator and never
// written to global memory.
#define TILE_H 8
#define TILE_W 8
#define CHUNK  32
extern __shared__ float smem[];   // [TILE_H*TILE_W*Cout] acc  ||  halo reused after

__global__ void fused_dwpw(const __half* __restrict__ in,   // [N,H,W,C]
                           const __half* __restrict__ dwW,  // [C,3,3]
                           const __half* __restrict__ pwW,  // [Cout,C]
                           __half* __restrict__ out,        // [N,H,W,Cout]
                           int N,int C,int Cout,int H,int W){
  const int tile_x = blockIdx.x*TILE_W, tile_y = blockIdx.y*TILE_H, n = blockIdx.z;
  const int tx = threadIdx.x, ty = threadIdx.y;     // pixel within tile
  const int ox = tile_x+tx, oy = tile_y+ty;
  const int npix = TILE_H*TILE_W;
  float* acc = smem;                                 // [npix*Cout]
  __half* halo = (__half*)(smem + npix*Cout);        // [(TILE_H+2)*(TILE_W+2)*CHUNK]
  const int HW=(TILE_H+2), WW=(TILE_W+2);
  const int pix = ty*TILE_W+tx;
  for(int co=0; co<Cout; ++co) acc[pix*Cout+co]=0.f;

  for(int cc=0; cc<C; cc+=CHUNK){
    int chunk = min(CHUNK, C-cc);
    // cooperative halo load: (TILE_H+2)x(TILE_W+2) x chunk, centered with -1 pad
    for(int idx=ty*TILE_W+tx; idx<HW*WW; idx+=npix){
      int ly=idx/WW, lx=idx%WW;
      int gy=tile_y+ly-1, gx=tile_x+lx-1;
      bool in_b = (gy>=0&&gy<H&&gx>=0&&gx<W);
      const __half* src = in + (((long)n*H+gy)*W+gx)*C + cc;
      for(int c=0;c<chunk;++c)
        halo[(ly*WW+lx)*CHUNK+c] = in_b ? src[c] : __float2half(0.f);
    }
    __syncthreads();
    if(ox<W && oy<H){
      for(int c=0;c<chunk;++c){
        // depthwise 3x3 at (ty,tx) for channel cc+c
        float dw=0.f;
        #pragma unroll
        for(int i=0;i<3;++i)
          #pragma unroll
          for(int j=0;j<3;++j)
            dw += __half2float(halo[((ty+i)*WW+(tx+j))*CHUNK+c])
                * __half2float(dwW[(cc+c)*9 + i*3+j]);
        dw = dw>0.f?dw:0.f;                           // ReLU on depthwise out
        // pointwise: scatter into all Cout accumulators
        const __half* pwcol = pwW + (cc+c);           // stride Cout
        for(int co=0; co<Cout; ++co)
          acc[pix*Cout+co] += dw * __half2float(pwW[co*C + (cc+c)]);
      }
    }
    __syncthreads();
  }
  if(ox<W && oy<H){
    __half* dst = out + (((long)n*H+oy)*W+ox)*Cout;
    for(int co=0; co<Cout; ++co) dst[co]=__float2half(acc[pix*Cout+co]);
  }
}

// ---- WMMA tensor-core fused kernel ----------------------------------------
// The corrected design: depthwise is a PROLOGUE that produces the GEMM A-matrix
// (dw_out[16 spatial][C]) in shared memory; the pointwise 1x1 then runs as a
// real tensor-core GEMM (WMMA 16x16x16) over K=C — so the compute-bound half
// stays on tensor cores AND the depthwise intermediate never hits DRAM.
// One warp/block handles 16 contiguous W positions at row y of image n.
// Requires C%16==0 and Cout%16==0 (PP-LCNetV3's 48/96/160/320 channels qualify).
using namespace nvcuda;
#define NWARPS 8
__global__ void fused_wmma(const __half* __restrict__ in,
                           const __half* __restrict__ dwW,
                           const __half* __restrict__ pwW,
                           __half* __restrict__ out,
                           int N,int C,int Cout,int H,int W){
  extern __shared__ __half sh[];
  const int COLS=18;                       // 16 + 2 halo
  __half* halo = sh;                       // [3][COLS][C]
  __half* As   = sh + 3*COLS*C;            // [16][C]  (dw_out, row-major)
  float*  Cs   = (float*)(As + 16*C);      // [NWARPS][16][16] per-warp output stage
  const int tid = threadIdx.x;             // 0..NWARPS*32-1
  const int warp = tid>>5, lane = tid&31;
  const int x0 = blockIdx.x*16, y = blockIdx.y, n = blockIdx.z;

  // 1. load halo (all block threads cooperate)
  for(int idx=tid; idx<3*COLS*C; idx+=NWARPS*32){
    int c = idx % C; int t = idx / C; int col=t%COLS, r=t/COLS;
    int gy=y+r-1, gx=x0+col-1;
    bool ok=(gy>=0&&gy<H&&gx>=0&&gx<W);
    halo[idx] = ok ? in[(((long)n*H+gy)*W+gx)*C + c] : __float2half(0.f);
  }
  __syncthreads();
  // 2. depthwise prologue (all block threads cooperate)
  for(int idx=tid; idx<16*C; idx+=NWARPS*32){
    int c=idx%C, m=idx/C;
    float acc=0.f;
    #pragma unroll
    for(int i=0;i<3;++i)
      #pragma unroll
      for(int j=0;j<3;++j)
        acc += __half2float(halo[((i*COLS)+(m+j))*C + c]) * __half2float(dwW[c*9+i*3+j]);
    As[m*C+c] = __float2half(acc>0.f?acc:0.f);
  }
  __syncthreads();
  // 3. pointwise GEMM, output-channel tiles distributed across warps
  float* myCs = Cs + warp*256;
  for(int n0=warp*16; n0<Cout; n0+=NWARPS*16){
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<C;k0+=16){
      wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
      wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
      wmma::load_matrix_sync(a, As + k0, C);
      wmma::load_matrix_sync(b, pwW + n0*C + k0, C);
      wmma::mma_sync(acc,a,b,acc);
    }
    wmma::store_matrix_sync(myCs, acc, 16, wmma::mem_row_major);
    __syncwarp();
    for(int idx=lane; idx<16*16; idx+=32){
      int nn=idx%16, m=idx/16; int gx=x0+m;
      if(gx<W) out[(((long)n*H+y)*W+gx)*Cout + n0+nn] = __float2half(myCs[m*16+nn]);
    }
    __syncwarp();
  }
}

static void fill(std::vector<__half>&v,std::mt19937&r){
  std::uniform_real_distribution<float> d(-0.3f,0.3f);
  for(auto&x:v) x=__float2half(d(r));
}

int main(int argc,char**argv){
  int N=argc>1?atoi(argv[1]):1, C=argc>2?atoi(argv[2]):96,
      Cout=argc>3?atoi(argv[3]):96, H=argc>4?atoi(argv[4]):120, W=argc>5?atoi(argv[5]):120;
  printf("Block: N=%d C=%d Cout=%d H=%d W=%d (NHWC, FP16, DW3x3+ReLU -> PW1x1)\n",N,C,Cout,H,W);
  std::mt19937 rng(123);
  std::vector<__half> h_in((long)N*H*W*C), h_dw((long)C*9), h_pw((long)Cout*C);
  fill(h_in,rng); fill(h_dw,rng); fill(h_pw,rng);
  __half *d_in,*d_dw,*d_pw,*d_out,*d_mid;
  CUDA_CHECK(cudaMalloc(&d_in,h_in.size()*2)); CUDA_CHECK(cudaMalloc(&d_dw,h_dw.size()*2));
  CUDA_CHECK(cudaMalloc(&d_pw,h_pw.size()*2));
  CUDA_CHECK(cudaMalloc(&d_out,(long)N*H*W*Cout*2)); CUDA_CHECK(cudaMalloc(&d_mid,(long)N*H*W*C*2));
  CUDA_CHECK(cudaMemcpy(d_in,h_in.data(),h_in.size()*2,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dw,h_dw.data(),h_dw.size()*2,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pw,h_pw.data(),h_pw.size()*2,cudaMemcpyHostToDevice));

  // ---- cuDNN baseline: depthwise conv (groups=C) then pointwise 1x1 conv ----
  cudnnHandle_t cu; CUDNN_CHECK(cudnnCreate(&cu));
  auto mkT=[&](int c){ cudnnTensorDescriptor_t t; CUDNN_CHECK(cudnnCreateTensorDescriptor(&t));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(t,CUDNN_TENSOR_NHWC,CUDNN_DATA_HALF,N,c,H,W)); return t; };
  cudnnTensorDescriptor_t tIn=mkT(C), tMid=mkT(C), tOut=mkT(Cout);
  cudnnFilterDescriptor_t fDw,fPw;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&fDw)); CUDNN_CHECK(cudnnCreateFilterDescriptor(&fPw));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(fDw,CUDNN_DATA_HALF,CUDNN_TENSOR_NHWC,C,1,3,3));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(fPw,CUDNN_DATA_HALF,CUDNN_TENSOR_NHWC,Cout,C,1,1));
  cudnnConvolutionDescriptor_t cDw,cPw;
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&cDw)); CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&cPw));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(cDw,1,1,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetConvolutionGroupCount(cDw,C));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(cPw,0,0,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetConvolutionMathType(cDw,CUDNN_TENSOR_OP_MATH));
  CUDNN_CHECK(cudnnSetConvolutionMathType(cPw,CUDNN_TENSOR_OP_MATH));
  size_t ws=0; void* dws=nullptr;
  cudnnConvolutionFwdAlgo_t aDw=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                            aPw=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  size_t w1=0,w2=0;
  cudnnGetConvolutionForwardWorkspaceSize(cu,tIn,fDw,cDw,tMid,aDw,&w1);
  cudnnGetConvolutionForwardWorkspaceSize(cu,tMid,fPw,cPw,tOut,aPw,&w2);
  ws=w1>w2?w1:w2; if(ws) CUDA_CHECK(cudaMalloc(&dws,ws));
  cudaStream_t strm; CUDA_CHECK(cudaStreamCreate(&strm));
  CUDNN_CHECK(cudnnSetStream(cu,strm));
  const float one=1.f,zero=0.f;
  auto runCudnn=[&](){
    cudnnConvolutionForward(cu,&one,tIn,d_in,fDw,d_dw,cDw,aDw,dws,ws,&zero,tMid,d_mid);
    cudnnConvolutionForward(cu,&one,tMid,d_mid,fPw,d_pw,cPw,aPw,dws,ws,&zero,tOut,d_out);
  };
  // ---- fused launch ----
  dim3 blk(TILE_W,TILE_H), grd((W+TILE_W-1)/TILE_W,(H+TILE_H-1)/TILE_H,N);
  size_t shmem = (size_t)TILE_H*TILE_W*Cout*sizeof(float)
               + (size_t)(TILE_H+2)*(TILE_W+2)*CHUNK*sizeof(__half);
  CUDA_CHECK(cudaFuncSetAttribute(fused_dwpw,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem));
  auto runFused=[&](){ fused_dwpw<<<grd,blk,shmem,strm>>>(d_in,d_dw,d_pw,d_out,N,C,Cout,H,W); };

  // WMMA fused (tensor-core PW). Needs C%16==0 && Cout%16==0.
  __half* d_outW=nullptr; bool wmma_ok=(C%16==0&&Cout%16==0);
  size_t shW = (size_t)3*18*C*2 + (size_t)16*C*2 + (size_t)NWARPS*16*16*4;
  dim3 blkW(NWARPS*32), grdW((W+15)/16, H, N);
  if(wmma_ok){
    CUDA_CHECK(cudaMalloc(&d_outW,(long)N*H*W*Cout*2));
    CUDA_CHECK(cudaFuncSetAttribute(fused_wmma,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shW));
  }
  auto runWmma=[&](){ fused_wmma<<<grdW,blkW,shW,strm>>>(d_in,d_dw,d_pw,d_outW,N,C,Cout,H,W); };

  printf("shared mem/block: scalar=%.1f KB  wmma=%.1f KB%s\n", shmem/1024.0, shW/1024.0,
         wmma_ok?"":"  (WMMA SKIPPED: C/Cout not mult of 16)");
  // correctness: fused vs cuDNN (note: ReLU on dw in fused; replicate by hand-check magnitude)
  // time both
  cudaEvent_t ev_s,ev_e; cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
  // graph-captured timing: removes per-launch CPU overhead from BOTH paths
  // (mirrors how a TRT engine runs under CUDA-graph capture). This is the fair
  // compute+memory comparison — if the fused kernel still wins here, it's real.
  auto timeit=[&](auto fn,const char*name)->float{
    for(int i=0;i<30;++i) fn(); CUDA_CHECK(cudaStreamSynchronize(strm));
    cudaGraph_t g; cudaGraphExec_t ge;
    cudaError_t cerr=cudaStreamBeginCapture(strm,cudaStreamCaptureModeThreadLocal);
    if(cerr!=cudaSuccess){ printf("%s capture-begin err: %s\n",name,cudaGetErrorString(cerr)); return -1; }
    fn();
    cerr=cudaStreamEndCapture(strm,&g);
    if(cerr!=cudaSuccess){ printf("%s capture-end err (not graph-safe): %s\n",name,cudaGetErrorString(cerr)); return -1; }
    CUDA_CHECK(cudaGraphInstantiate(&ge,g,0));
    CUDA_CHECK(cudaEventRecord(ev_s,strm));
    for(int i=0;i<300;++i) CUDA_CHECK(cudaGraphLaunch(ge,strm));
    CUDA_CHECK(cudaEventRecord(ev_e,strm));
    CUDA_CHECK(cudaEventSynchronize(ev_e)); float ms; cudaEventElapsedTime(&ms,ev_s,ev_e);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    return ms/300.f; };
  float tC=timeit(runCudnn,"cudnn"), tF=timeit(runFused,"fused");
  float tW = wmma_ok ? timeit(runWmma,"wmma") : 0.f;
  // correctness: WMMA vs scalar fused (same math: ReLU-dw + PW)
  double maxdiff=-1;
  if(wmma_ok){
    runFused(); std::vector<__half> a((long)N*H*W*Cout); CUDA_CHECK(cudaMemcpy(a.data(),d_out,a.size()*2,cudaMemcpyDeviceToHost));
    runWmma();  std::vector<__half> b((long)N*H*W*Cout); CUDA_CHECK(cudaMemcpy(b.data(),d_outW,b.size()*2,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<a.size();++i){ double d=fabs(__half2float(a[i])-__half2float(b[i])); if(d>maxdiff)maxdiff=d; }
  }

  // DRAM traffic model (bytes), per call
  double in_b=(double)N*H*W*C*2, mid_b=(double)N*H*W*C*2, out_b=(double)N*H*W*Cout*2;
  double base_dram = in_b + mid_b /*write*/ + mid_b /*read*/ + out_b;
  double fused_dram= in_b + out_b;   // intermediate stays on-chip
  printf("\ncuDNN  (DW+PW separate): %.4f ms\n",tC);
  printf("scalar (DW->PW on-chip): %.4f ms   speedup=%.2fx\n",tF,tC/tF);
  if(wmma_ok) printf("WMMA   (DW prologue + TC GEMM PW): %.4f ms   speedup=%.2fx   maxdiff(vs scalar)=%.3g\n",
                     tW, tC/tW, maxdiff);
  printf("DRAM model: baseline=%.2f MB  fused=%.2f MB  (%.0f%% traffic cut)\n",
         base_dram/1e6, fused_dram/1e6, 100*(1-fused_dram/base_dram));
  return 0;
}
