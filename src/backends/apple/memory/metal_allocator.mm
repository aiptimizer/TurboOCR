// MetalAllocator implementation (see metal_allocator.h).

#import "apple/memory/metal_allocator.h"
#import "apple/support/metal_common.h"
#import "apple/memory/metal_image.h" // unregister_texture
#import "apple/support/apple_contention.h"

#include <cstdlib>
#include <cstring>

namespace turbo_ocr::apple {

void *MetalAllocator::allocate(std::size_t bytes) {
  if (bytes == 0) bytes = 1;
  TURBO_APPLE_STAT_N(alloc_new_buffer, bytes);
  id<MTLBuffer> buf =
      [mtl_device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (!buf) return nullptr;
  register_buffer(buf); // keep the id<MTLBuffer> alive + resolvable by .contents
  // NB: register_buffer holds a strong ref in the registry; the ObjC object is
  // released in free() when we unregister.
  return buf.contents;
}

void MetalAllocator::free(void *p) noexcept {
  if (!p) return;
  TURBO_APPLE_STAT(alloc_free_buffer);
  // BOTH registries are keyed by this pointer and BOTH must be dropped here.
  //
  // ensure_texture() lazily builds an RGBA8 sampler texture for a page buffer
  // and registers it under the buffer's .contents address (metal_image.mm). Only
  // ~MetalImage used to unregister it — but UnifiedOcrPipeline::upload_image_
  // never constructs a MetalImage, it allocates straight through this allocator.
  // So every page leaked a ~3 MB texture forever AND, once Metal recycled a
  // freed .contents address for the next page, texture_for() returned the
  // PREVIOUS page's pixels to resize_normalize/warp_crops — silent wrong-page
  // detection and recognition that gets more likely the more pages are in
  // flight. Unregistering here closes both.
  unregister_texture(p);
  unregister_buffer(p); // drops the registry's strong ref -> MTLBuffer released
}

void *MetalAllocator::allocate_host(std::size_t bytes) {
  return std::malloc(bytes ? bytes : 1);
}

void MetalAllocator::free_host(void *p) noexcept { std::free(p); }

void MetalAllocator::copy_h2d(void *dst, const void *src, std::size_t bytes,
                              backend::DeviceQueue &) {
  TURBO_APPLE_STAT_N(alloc_h2d, bytes);
  std::memcpy(dst, src, bytes); // unified memory: coherent
  // New page pixels in this buffer => any cached sampler texture for it is stale.
  // Required now that the shared pipeline REUSES one upload buffer per replica
  // (see UnifiedOcrPipeline::upload_image_): the key no longer changes per page.
  invalidate_texture(dst);
}

void MetalAllocator::copy_d2h(void *dst, const void *src, std::size_t bytes,
                              backend::DeviceQueue &) {
  std::memcpy(dst, src, bytes);
}

void MetalAllocator::copy_d2d(void *dst, const void *src, std::size_t bytes,
                              backend::DeviceQueue &) {
  std::memcpy(dst, src, bytes);
}

std::shared_ptr<MetalAllocator> shared_allocator() {
  static std::shared_ptr<MetalAllocator> a = std::make_shared<MetalAllocator>();
  return a;
}

} // namespace turbo_ocr::apple
