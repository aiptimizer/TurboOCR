// SYNTAX-CHECK-ONLY HIP stub — see ../cuda_runtime.h in this directory.
#pragma once
#include <cstddef>
typedef struct ihipStream_t *hipStream_t;
typedef struct ihipEvent_t *hipEvent_t;
typedef struct ihipGraph *hipGraph_t;
typedef struct ihipGraphExec *hipGraphExec_t;
typedef enum { hipSuccess = 0, hipErrorInvalidValue, hipErrorOutOfMemory,
               hipErrorNotReady, hipErrorIllegalAddress, hipErrorLaunchFailure,
               hipErrorLaunchTimeOut, hipErrorECCNotCorrectable = 214,
               hipErrorContextIsDestroyed = 709,
               hipErrorUnknown = 999 } hipError_t;
typedef enum { hipMemcpyHostToHost, hipMemcpyHostToDevice, hipMemcpyDeviceToHost,
               hipMemcpyDeviceToDevice, hipMemcpyDefault } hipMemcpyKind;
typedef enum { hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal,
               hipStreamCaptureModeRelaxed } hipStreamCaptureMode;
struct hipDeviceProp_t { char name[256]; size_t totalGlobalMem; int major, minor;
                         int multiProcessorCount; char gcnArchName[256]; };
#define hipStreamNonBlocking 0x01
#define hipEventDisableTiming 0x02
#define hipHostMallocDefault 0x00
extern "C" {
hipError_t hipMalloc(void **, size_t);
hipError_t hipFree(void *);
hipError_t hipHostMalloc(void **, size_t, unsigned);
hipError_t hipHostFree(void *);
hipError_t hipMemcpy(void *, const void *, size_t, hipMemcpyKind);
hipError_t hipMemcpyAsync(void *, const void *, size_t, hipMemcpyKind, hipStream_t);
// Pitched 2-D copy — the HIP twin of cudaMemcpy2DAsync. Used by
// amd/kernels_hip/hip_kernels.cpp to upload a non-continuous cv::Mat (row
// stride != cols*3) without a host repack.
hipError_t hipMemcpy2D(void *, size_t, const void *, size_t, size_t, size_t, hipMemcpyKind);
hipError_t hipMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t,
                            hipMemcpyKind, hipStream_t);
hipError_t hipMemset(void *, int, size_t);
hipError_t hipMemsetAsync(void *, int, size_t, hipStream_t);
hipError_t hipStreamCreateWithFlags(hipStream_t *, unsigned);
hipError_t hipStreamCreate(hipStream_t *);
hipError_t hipStreamDestroy(hipStream_t);
hipError_t hipStreamSynchronize(hipStream_t);
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned);
hipError_t hipStreamQuery(hipStream_t);
hipError_t hipStreamBeginCapture(hipStream_t, hipStreamCaptureMode);
hipError_t hipStreamEndCapture(hipStream_t, hipGraph_t *);
hipError_t hipGraphInstantiate(hipGraphExec_t *, hipGraph_t, void *, char *, size_t);
hipError_t hipGraphLaunch(hipGraphExec_t, hipStream_t);
hipError_t hipGraphExecDestroy(hipGraphExec_t);
hipError_t hipGraphDestroy(hipGraph_t);
hipError_t hipEventCreateWithFlags(hipEvent_t *, unsigned);
hipError_t hipEventCreate(hipEvent_t *);
hipError_t hipEventDestroy(hipEvent_t);
hipError_t hipEventRecord(hipEvent_t, hipStream_t);
hipError_t hipEventSynchronize(hipEvent_t);
hipError_t hipEventQuery(hipEvent_t);
hipError_t hipEventElapsedTime(float *, hipEvent_t, hipEvent_t);
hipError_t hipGetDeviceCount(int *);
hipError_t hipSetDevice(int);
hipError_t hipGetDevice(int *);
hipError_t hipGetDeviceProperties(hipDeviceProp_t *, int);
hipError_t hipDriverGetVersion(int *);
hipError_t hipRuntimeGetVersion(int *);
hipError_t hipDeviceSynchronize(void);
hipError_t hipGetLastError(void);
hipError_t hipPeekAtLastError(void);
hipError_t hipMemGetInfo(size_t *, size_t *);
const char *hipGetErrorString(hipError_t);
const char *hipGetErrorName(hipError_t);
}
