// SYNTAX-CHECK-ONLY MIGraphX stub — see ../cuda_runtime.h in this directory.
#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "hip/hip_runtime.h"
typedef enum {
  migraphx_shape_tuple_type = 0, migraphx_shape_bool_type, migraphx_shape_half_type,
  migraphx_shape_float_type, migraphx_shape_double_type, migraphx_shape_uint8_type,
  migraphx_shape_int8_type, migraphx_shape_uint16_type, migraphx_shape_int16_type,
  migraphx_shape_int32_type, migraphx_shape_int64_type, migraphx_shape_uint32_type,
  migraphx_shape_uint64_type, migraphx_shape_fp8e4m3fnuz_type
} migraphx_shape_datatype_t;
namespace migraphx {
struct shape {
  std::vector<std::size_t> lengths() const;
  std::vector<std::size_t> strides() const;
  std::size_t bytes() const;
  std::size_t elements() const;
  migraphx_shape_datatype_t type() const;
};
struct argument {
  argument() = default;
  argument(const shape &, void *);
  shape get_shape() const;
  char *data() const;
};
// NO default constructor, matching the real header (a handle type only ever
// produced by an eval) — verified against ROCm 7.1.1 on 2026-08-02, where the
// shim's earlier default-constructibility hid a real compile error.
struct arguments {
  arguments() = delete;
  arguments(const arguments &);
  arguments &operator=(const arguments &);
  std::size_t size() const;
  argument operator[](std::size_t) const;
};
struct program_parameter_shapes {
  shape operator[](const char *) const;
  std::vector<const char *> names() const;
  std::size_t size() const;
};
struct program_parameters {
  void add(const char *, const argument &);
};
struct onnx_options {
  void set_input_parameter_shape(const std::string &, std::vector<std::size_t>);
  void set_default_dim_value(std::size_t);
};
struct target { target() = default; explicit target(const char *); };
struct compile_options { bool offload_copy = false; bool fast_math = true;
                         void set_offload_copy(bool = true);
                         void set_fast_math(bool = true); };
struct program {
  void compile(const target &, const compile_options & = {});
  program_parameter_shapes get_parameter_shapes() const;
  arguments eval(const program_parameters &) const;
  std::vector<shape> get_output_shapes() const;
  // Real signature (ROCm 7.1.1): a const member template over the stream's
  // pointee type; the type name is stringified internally.
  template <class Stream>
  arguments run_async(const program_parameters &, Stream *) const;
};
program parse_onnx(const char *, const onnx_options & = {});
program load(const char *);
void save(const program &, const char *);
void quantize_fp16(program &);
void quantize_int8(program &);
} // namespace migraphx
