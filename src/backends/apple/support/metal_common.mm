// Apple backend — shared Metal plumbing implementation (see metal_common.h).

#import "apple/support/metal_common.h"
#import "apple/support/apple_contention.h"

#import <Foundation/Foundation.h>

#include "turbo_ocr/base/env_utils.h"

#include <dlfcn.h>

#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace turbo_ocr::apple {

// --- device / library singletons -------------------------------------------

id<MTLDevice> mtl_device() {
  static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  return dev;
}

id<MTLCommandQueue> mtl_command_queue() {
  static id<MTLCommandQueue> q = [mtl_device() newCommandQueue];
  return q;
}

MPSGraphDevice *mps_graph_device() {
  static MPSGraphDevice *g = [MPSGraphDevice deviceWithMTLDevice:mtl_device()];
  return g;
}

const char *metallib_path() {
  // Held in a function-local static, not returned straight out of the
  // environment block: the returned pointer outlives this call and a later
  // setenv() may reallocate the block underneath it.
  static const std::string override_path = env::env_or("TURBO_APPLE_METALLIB", "");
  if (!override_path.empty()) return override_path.c_str();
  static std::string cached;
  if (!cached.empty()) return cached.c_str();
  @autoreleasepool {
    NSMutableArray<NSString *> *dirs = [NSMutableArray array];
    // Alongside the executable is the server install layout.
    NSString *exe = [[NSBundle mainBundle] executablePath];
    if (exe) [dirs addObject:[exe stringByDeletingLastPathComponent]];
    // Alongside THIS image is the Python-wheel layout: the backend is linked
    // into _turboocr.so inside site-packages/turboocr/, and the wheel ships
    // the metallib next to it. dladdr on our own symbol finds that image
    // whether we live in a dylib/extension or the main executable.
    Dl_info info;
    if (dladdr(reinterpret_cast<const void *>(&metallib_path), &info) &&
        info.dli_fname) {
      NSString *self_dir = [[NSString stringWithUTF8String:info.dli_fname]
          stringByDeletingLastPathComponent];
      if (self_dir && ![dirs containsObject:self_dir]) [dirs addObject:self_dir];
    }
    for (NSString *dir in dirs) {
      NSString *cand = [dir stringByAppendingPathComponent:@"turbo_apple.metallib"];
      if ([[NSFileManager defaultManager] fileExistsAtPath:cand]) {
        cached = cand.UTF8String;
        return cached.c_str();
      }
    }
  }
  cached = "turbo_apple.metallib"; // last resort: CWD
  return cached.c_str();
}

id<MTLLibrary> mtl_library() {
  // If initialization throws, the static is retried on the next call
  // (per [stmt.dcl]) — a fixed environment does not need a process restart.
  static id<MTLLibrary> lib = ^id<MTLLibrary>() {
    @autoreleasepool {
      NSError *err = nil;
      NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path()]];
      id<MTLLibrary> l = [mtl_device() newLibraryWithURL:url error:&err];
      if (!l) {
        // Throw rather than return nil: no caller null-checks the pipeline
        // states built from this library, and Metal segfaults on a nil PSO.
        // A missing shader bundle must surface as a catchable load error.
        throw std::runtime_error(
            std::string("[apple] failed to load Metal shader library at '") +
            metallib_path() +
            "' — set TURBO_APPLE_METALLIB or place turbo_apple.metallib next "
            "to the executable/module (" +
            (err ? err.localizedDescription.UTF8String : "no NSError") + ")");
      }
      return l;
    }
  }();
  return lib;
}

id<MTLComputePipelineState> mtl_pipeline(const char *fn_name) {
  static std::mutex m;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  TURBO_APPLE_STAT(pso_lookup);
  TURBO_APPLE_LOCK(pso_lock, m);
  auto it = cache.find(fn_name);
  if (it != cache.end()) return it->second;
  @autoreleasepool {
    NSError *err = nil;
    id<MTLFunction> fn = [mtl_library() newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
    // Throw, don't return nil — see mtl_library(). A missing function means a
    // stale/mismatched metallib, which must be a load error, not a segfault.
    if (!fn)
      throw std::runtime_error(std::string("[apple] shader function '") +
                               fn_name + "' missing from " + metallib_path());
    id<MTLComputePipelineState> pso =
        [mtl_device() newComputePipelineStateWithFunction:fn error:&err];
    if (!pso)
      throw std::runtime_error(
          std::string("[apple] compute pipeline for '") + fn_name + "' failed: " +
          (err ? err.localizedDescription.UTF8String : "unknown Metal error"));
    cache[fn_name] = pso;
    return pso;
  }
}

// --- void* <-> MTLBuffer registry ------------------------------------------
// Sorted by base address so resolve() finds the containing buffer with an
// upper_bound + range check (supports pointers offset into a buffer).

namespace {
struct Reg {
  std::mutex m;
  std::map<std::uintptr_t, std::pair<id<MTLBuffer>, std::size_t>> by_base; // base -> {buf,len}
};
Reg &reg() {
  static Reg r;
  return r;
}
} // namespace

void register_buffer(id<MTLBuffer> buf) {
  if (!buf || !buf.contents) return;
  auto &r = reg();
  TURBO_APPLE_LOCK(reg_register_lock, r.m);
  r.by_base[reinterpret_cast<std::uintptr_t>(buf.contents)] = {buf, (std::size_t)buf.length};
}

void unregister_buffer(void *contents) {
  if (!contents) return;
  auto &r = reg();
  TURBO_APPLE_LOCK(reg_unregister_lock, r.m);
  r.by_base.erase(reinterpret_cast<std::uintptr_t>(contents));
}

id<MTLBuffer> resolve_buffer(const void *p, std::size_t *offset_out) {
  if (!p) return nil;
  auto &r = reg();
  TURBO_APPLE_STAT(reg_resolve);
  TURBO_APPLE_LOCK(reg_resolve_lock, r.m);
  const auto addr = reinterpret_cast<std::uintptr_t>(p);
  // First base > addr, then step back one to the candidate containing buffer.
  auto it = r.by_base.upper_bound(addr);
  if (it == r.by_base.begin()) return nil;
  --it;
  const std::uintptr_t base = it->first;
  const std::size_t len = it->second.second;
  if (addr < base || addr >= base + len) return nil;
  if (offset_out) *offset_out = addr - base;
  return it->second.first;
}

} // namespace turbo_ocr::apple
