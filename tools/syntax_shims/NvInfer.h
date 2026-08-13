// SYNTAX-CHECK-ONLY TensorRT stub — see cuda_runtime.h in this directory.
#pragma once
#include <cstdint>
#include <cstddef>
#include "cuda_runtime.h"
namespace nvinfer1 {
class ILogger {
public:
  enum class Severity : int32_t { kINTERNAL_ERROR=0, kERROR=1, kWARNING=2, kINFO=3, kVERBOSE=4 };
  virtual void log(Severity, const char *) noexcept = 0;
  virtual ~ILogger() = default;
};
static constexpr int32_t kMAX_DIMS = 8;
struct Dims { static constexpr int32_t MAX_DIMS = kMAX_DIMS;
              int32_t nbDims = 0; int64_t d[kMAX_DIMS] = {}; };
using Dims4 = Dims; using Dims2 = Dims; using Dims3 = Dims;
enum class DataType : int32_t { kFLOAT=0, kHALF=1, kINT8=2, kINT32=3, kBOOL=4, kUINT8=5 };
enum class TensorIOMode : int32_t { kNONE=0, kINPUT=1, kOUTPUT=2 };
enum class OptProfileSelector : int32_t { kMIN=0, kOPT=1, kMAX=2 };
// Declared here rather than with the other builder-side enums below because
// ICudaEngine::createExecutionContext takes it.
enum class ExecutionContextAllocationStrategy : int32_t {
  kSTATIC=0, kON_PROFILE_CHANGE=1, kUSER_MANAGED=2 };
class IExecutionContext {
public:
  virtual ~IExecutionContext() = default;
  bool setInputShape(const char *, const Dims &) noexcept;
  bool setTensorAddress(const char *, void *) noexcept;
  bool setInputTensorAddress(const char *, const void *) noexcept;
  bool enqueueV3(cudaStream_t) noexcept;
  bool allInputDimensionsSpecified() const noexcept;
  Dims getTensorShape(const char *) const noexcept;
  // Returns bool in TensorRT, and trt_engine.cpp branches on it to avoid baking
  // a CUDA graph bound to the wrong profile. The stub said void, which nothing
  // caught while trt_engine.cpp sat outside the gate.
  bool setOptimizationProfileAsync(int32_t, cudaStream_t) noexcept;
  void setDeviceMemoryV2(void *, int64_t) noexcept;
};
class ICudaEngine {
public:
  virtual ~ICudaEngine() = default;
  IExecutionContext *createExecutionContext() noexcept;
  IExecutionContext *
  createExecutionContext(ExecutionContextAllocationStrategy) noexcept;
  int32_t getNbIOTensors() const noexcept;
  const char *getIOTensorName(int32_t) const noexcept;
  TensorIOMode getTensorIOMode(const char *) const noexcept;
  DataType getTensorDataType(const char *) const noexcept;
  Dims getTensorShape(const char *) const noexcept;
  Dims getProfileShape(const char *, int32_t, OptProfileSelector) const noexcept;
  int32_t getNbOptimizationProfiles() const noexcept;
  int64_t getDeviceMemorySizeForProfileV2(int32_t) const noexcept;
};
class IRuntime {
public:
  virtual ~IRuntime() = default;
  ICudaEngine *deserializeCudaEngine(const void *, std::size_t) noexcept;
};
IRuntime *createInferRuntime(ILogger &) noexcept;

// ---- BUILDER SIDE --------------------------------------------------------
// Only the runtime half of TensorRT was stubbed, so onnx_to_trt.cpp,
// trt_profiles.cpp, trt_engine.cpp, trt_engine_cache.cpp and engine_loader.cpp
// were type-checked by nothing on any platform — they compile only in a real
// CUDA+TRT configure, which no gate machine has. Everything below exists to
// close that hole; signatures match TensorRT 10 closely enough to catch a
// misuse, not to model behaviour.
enum class BuilderFlag : int32_t { kFP16=0, kINT8=1, kFP8=2, kBF16=3, kREFIT=4 };
enum class MemoryPoolType : int32_t { kWORKSPACE=0, kDLA_MANAGED_SRAM=1,
                                      kTACTIC_DRAM=2 };
enum class NetworkDefinitionCreationFlag : int32_t { kEXPLICIT_BATCH=0,
                                                     kSTRONGLY_TYPED=1 };
class ITensor {
public:
  virtual ~ITensor() = default;
  const char *getName() const noexcept;
  Dims getDimensions() const noexcept;
  DataType getType() const noexcept;
};
class IHostMemory {
public:
  virtual ~IHostMemory() = default;
  void *data() const noexcept;
  std::size_t size() const noexcept;
};
class INetworkDefinition {
public:
  virtual ~INetworkDefinition() = default;
  int32_t getNbInputs() const noexcept;
  ITensor *getInput(int32_t) const noexcept;
  int32_t getNbOutputs() const noexcept;
  ITensor *getOutput(int32_t) const noexcept;
};
class IOptimizationProfile {
public:
  virtual ~IOptimizationProfile() = default;
  bool setDimensions(const char *, OptProfileSelector, const Dims &) noexcept;
};
class IBuilderConfig {
public:
  virtual ~IBuilderConfig() = default;
  void setMemoryPoolLimit(MemoryPoolType, std::size_t) noexcept;
  void setFlag(BuilderFlag) noexcept;
  void setBuilderOptimizationLevel(int32_t) noexcept;
  int32_t addOptimizationProfile(IOptimizationProfile *) noexcept;
  void setMaxAuxStreams(int32_t) noexcept;
};
class IBuilder {
public:
  virtual ~IBuilder() = default;
  INetworkDefinition *createNetworkV2(uint32_t) noexcept;
  IBuilderConfig *createBuilderConfig() noexcept;
  IOptimizationProfile *createOptimizationProfile() noexcept;
  IHostMemory *buildSerializedNetwork(INetworkDefinition &,
                                      IBuilderConfig &) noexcept;
};
IBuilder *createInferBuilder(ILogger &) noexcept;
} // namespace nvinfer1
