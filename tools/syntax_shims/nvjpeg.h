// SYNTAX-CHECK-ONLY nvJPEG stub — see cuda_runtime.h in this directory.
#pragma once
#include <cstddef>
#include "cuda_runtime.h"
#define NVJPEG_MAX_COMPONENT 4
typedef enum { NVJPEG_STATUS_SUCCESS = 0, NVJPEG_STATUS_NOT_INITIALIZED,
               NVJPEG_STATUS_ARCH_MISMATCH, NVJPEG_STATUS_EXECUTION_FAILED,
               NVJPEG_STATUS_INTERNAL_ERROR } nvjpegStatus_t;
typedef enum { NVJPEG_BACKEND_DEFAULT = 0, NVJPEG_BACKEND_HYBRID,
               NVJPEG_BACKEND_GPU_HYBRID, NVJPEG_BACKEND_HARDWARE } nvjpegBackend_t;
typedef enum { NVJPEG_CSS_444 = 0, NVJPEG_CSS_422, NVJPEG_CSS_420,
               NVJPEG_CSS_440, NVJPEG_CSS_411, NVJPEG_CSS_410,
               NVJPEG_CSS_GRAY, NVJPEG_CSS_UNKNOWN = -1 } nvjpegChromaSubsampling_t;
typedef enum { NVJPEG_OUTPUT_UNCHANGED = 0, NVJPEG_OUTPUT_YUV, NVJPEG_OUTPUT_Y,
               NVJPEG_OUTPUT_RGB, NVJPEG_OUTPUT_BGR, NVJPEG_OUTPUT_RGBI,
               NVJPEG_OUTPUT_BGRI } nvjpegOutputFormat_t;
typedef enum { NVJPEG_INPUT_RGB = 0, NVJPEG_INPUT_BGR, NVJPEG_INPUT_RGBI,
               NVJPEG_INPUT_BGRI } nvjpegInputFormat_t;
typedef struct nvjpegHandle *nvjpegHandle_t;
typedef struct nvjpegJpegState *nvjpegJpegState_t;
typedef struct nvjpegEncoderState *nvjpegEncoderState_t;
typedef struct nvjpegEncoderParams *nvjpegEncoderParams_t;
typedef struct { unsigned char *channel[NVJPEG_MAX_COMPONENT];
                 size_t pitch[NVJPEG_MAX_COMPONENT]; } nvjpegImage_t;
extern "C" {
nvjpegStatus_t nvjpegCreateSimple(nvjpegHandle_t *);
nvjpegStatus_t nvjpegCreateEx(nvjpegBackend_t, void *, void *, unsigned, nvjpegHandle_t *);
nvjpegStatus_t nvjpegDestroy(nvjpegHandle_t);
nvjpegStatus_t nvjpegJpegStateCreate(nvjpegHandle_t, nvjpegJpegState_t *);
nvjpegStatus_t nvjpegJpegStateDestroy(nvjpegJpegState_t);
nvjpegStatus_t nvjpegGetImageInfo(nvjpegHandle_t, const unsigned char *, size_t,
                                  int *, nvjpegChromaSubsampling_t *, int *, int *);
nvjpegStatus_t nvjpegDecode(nvjpegHandle_t, nvjpegJpegState_t, const unsigned char *,
                            size_t, nvjpegOutputFormat_t, nvjpegImage_t *, cudaStream_t);
nvjpegStatus_t nvjpegDecodeBatchedInitialize(nvjpegHandle_t, nvjpegJpegState_t, int, int,
                                             nvjpegOutputFormat_t);
nvjpegStatus_t nvjpegDecodeBatched(nvjpegHandle_t, nvjpegJpegState_t,
                                   const unsigned char *const *, const size_t *,
                                   nvjpegImage_t *, cudaStream_t);
// --- encoder -------------------------------------------------------------
// Added when src/image/page_image_encoder.cpp stopped including this arm
// directly: the nvJPEG *encoder* then reached a shim-checked TU for the first
// time (via cuda_backend.cpp) and the stub had only ever declared the decoder.
nvjpegStatus_t nvjpegEncoderStateCreate(nvjpegHandle_t, nvjpegEncoderState_t *, cudaStream_t);
nvjpegStatus_t nvjpegEncoderStateDestroy(nvjpegEncoderState_t);
nvjpegStatus_t nvjpegEncoderParamsCreate(nvjpegHandle_t, nvjpegEncoderParams_t *, cudaStream_t);
nvjpegStatus_t nvjpegEncoderParamsDestroy(nvjpegEncoderParams_t);
nvjpegStatus_t nvjpegEncoderParamsSetQuality(nvjpegEncoderParams_t, int, cudaStream_t);
nvjpegStatus_t nvjpegEncoderParamsSetSamplingFactors(nvjpegEncoderParams_t,
                                                     nvjpegChromaSubsampling_t, cudaStream_t);
nvjpegStatus_t nvjpegEncodeImage(nvjpegHandle_t, nvjpegEncoderState_t,
                                 nvjpegEncoderParams_t, const nvjpegImage_t *,
                                 nvjpegInputFormat_t, int, int, cudaStream_t);
nvjpegStatus_t nvjpegEncodeRetrieveBitstream(nvjpegHandle_t, nvjpegEncoderState_t,
                                             unsigned char *, size_t *, cudaStream_t);
}
