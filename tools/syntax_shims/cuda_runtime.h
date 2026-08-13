// SYNTAX-CHECK-ONLY CUDA stub. Not for compilation into any binary — it exists
// so the *_gpu.cpp HTTP routes can be type-checked on a machine with no CUDA
// toolkit, which is the only verification available for those files here.
#pragma once
#include <cstddef>
typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;
typedef struct CUgraph_st *cudaGraph_t;
typedef struct CUgraphExec_st *cudaGraphExec_t;
typedef enum { cudaStreamCaptureModeGlobal, cudaStreamCaptureModeThreadLocal,
               cudaStreamCaptureModeRelaxed } cudaStreamCaptureMode;
typedef enum { cudaSuccess = 0,
               cudaErrorIllegalAddress = 700,
               cudaErrorLaunchFailure = 701,
               cudaErrorLaunchTimeout = 702,
               cudaErrorHardwareStackError = 703,
               cudaErrorIllegalInstruction = 704,
               cudaErrorMisalignedAddress = 705,
               cudaErrorInvalidAddressSpace = 706,
               cudaErrorInvalidPc = 707,
               cudaErrorECCUncorrectable = 708,
               cudaErrorUnsupportedExecAffinity = 709,
               cudaErrorExternalDevice = 710,
               cudaErrorNvlinkUncorrectable = 711,
               cudaErrorMemoryAllocation = 712,
               cudaErrorInvalidValue = 713,
               cudaErrorNotReady = 714,
               cudaErrorInvalidDevice = 715,
               cudaErrorNoDevice = 716,
               cudaErrorInitializationError = 717,
               cudaErrorDeviceUninitialized = 718,
               cudaErrorContextIsDestroyed = 719,
               cudaErrorSystemNotReady = 720,
               cudaErrorSystemDriverMismatch = 721,
               cudaErrorCompatNotSupportedOnDevice = 722,
               cudaErrorInvalidPitchValue = 723,
               cudaErrorInvalidConfiguration = 724,
               cudaErrorUnknown = 999 } cudaError_t;
// cudaHostAlloc() flag bits. Real values from the CUDA runtime — these ARE the
// upstream constants, not fillers, because they are bit flags that get OR'd
// together and a wrong value would be a stub that disagrees with the real
// header (see README "Keeping the stubs honest").
#define cudaHostAllocDefault       0x00u
#define cudaHostAllocPortable      0x01u
#define cudaHostAllocMapped        0x02u
#define cudaHostAllocWriteCombined 0x04u
typedef enum { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
               cudaMemcpyDeviceToDevice, cudaMemcpyHostToHost,
               cudaMemcpyDefault } cudaMemcpyKind;
typedef enum { cudaDevAttrPageableMemoryAccess = 88,
               cudaDevAttrConcurrentManagedAccess = 89,
               cudaDevAttrUnifiedAddressing = 41,
               cudaDevAttrMemoryPoolsSupported = 115,
               // Used by trt_engine_cache.cpp to key a cached plan to the GPU
               // architecture and driver it was built for.
               cudaDevAttrComputeCapabilityMajor = 75,
               cudaDevAttrComputeCapabilityMinor = 76 } cudaDeviceAttr;
struct cudaDeviceProp { char name[256]; size_t totalGlobalMem; int major, minor;
                        int multiProcessorCount; };
extern "C" {
cudaError_t cudaMalloc(void **, size_t);
cudaError_t cudaFree(void *);
cudaError_t cudaMallocHost(void **, size_t);
cudaError_t cudaFreeHost(void *);
cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t);
cudaError_t cudaMemset(void *, int, size_t);
cudaError_t cudaStreamCreate(cudaStream_t *);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *, unsigned);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
cudaError_t cudaEventCreate(cudaEvent_t *);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventQuery(cudaEvent_t);
cudaError_t cudaStreamQuery(cudaStream_t);
cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
cudaError_t cudaGetDeviceCount(int *);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDevice(int *);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *, int);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetLastError(void);
cudaError_t cudaMallocAsync(void **, size_t, cudaStream_t);
cudaError_t cudaFreeAsync(void *, cudaStream_t);
cudaError_t cudaDeviceGetAttribute(int *, cudaDeviceAttr, int);
cudaError_t cudaDriverGetVersion(int *);
cudaError_t cudaRuntimeGetVersion(int *);
cudaError_t cudaHostRegister(void *, size_t, unsigned);
cudaError_t cudaHostUnregister(void *);
cudaError_t cudaHostAlloc(void **, size_t, unsigned);
cudaError_t cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t,
                              cudaMemcpyKind, cudaStream_t);
cudaError_t cudaMemcpy2D(void *, size_t, const void *, size_t, size_t, size_t,
                         cudaMemcpyKind);
cudaError_t cudaMallocPitch(void **, size_t *, size_t, size_t);
cudaError_t cudaPeekAtLastError(void);
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode);
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t *);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *, cudaGraph_t, unsigned long long);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t cudaGraphDestroy(cudaGraph_t);
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
cudaError_t cudaMemGetInfo(size_t *, size_t *);
const char *cudaGetErrorString(cudaError_t);
}
// Real cuda_runtime.h provides typed template overloads over the void** C API.
template <class T> cudaError_t cudaMallocHost(T **ptr, size_t size) {
  return cudaMallocHost(reinterpret_cast<void **>(ptr), size);
}
template <class T> cudaError_t cudaMalloc(T **ptr, size_t size) {
  return cudaMalloc(reinterpret_cast<void **>(ptr), size);
}
#define cudaStreamNonBlocking 0x01
#define cudaEventDisableTiming 0x02
