// SYNTAX-CHECK-ONLY TensorRT ONNX-parser stub — see cuda_runtime.h in this
// directory.
//
// Its absence is why src/backends/nvidia/engine/onnx_to_trt.cpp sat outside
// tools/syntax_shims/sources.txt: the file includes <NvOnnxParser.h>, nothing
// supplied it, so the one source that turns an ONNX export into a TRT plan was
// type-checked on no platform at all. Only what onnx_to_trt.cpp actually calls
// is modelled.
#pragma once
#include <cstdint>
#include "NvInfer.h"

namespace nvonnxparser {

// TensorRT spells the error accessor as a pointer to an immutable record; desc()
// is the human-readable text the caller prints per parse failure.
class IParserError {
public:
  virtual ~IParserError() = default;
  const char *desc() const noexcept;
  const char *file() const noexcept;
  int32_t line() const noexcept;
};

class IParser {
public:
  virtual ~IParser() = default;
  // The int is a nvinfer1::ILogger::Severity cast to int — TensorRT's own
  // signature takes a bare int here, so the call site's static_cast is real.
  bool parseFromFile(const char *, int) noexcept;
  bool parse(const void *, std::size_t, const char * = nullptr) noexcept;
  int32_t getNbErrors() const noexcept;
  const IParserError *getError(int32_t) const noexcept;
};

IParser *createParser(nvinfer1::INetworkDefinition &,
                      nvinfer1::ILogger &) noexcept;

} // namespace nvonnxparser
