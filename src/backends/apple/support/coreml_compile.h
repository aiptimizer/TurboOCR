#pragma once

// coreml_compile — process-wide .mlpackage -> .mlmodelc compile cache.
//
// CoreML loads only COMPILED models; compileModelAtURL: writes a temporary
// .mlmodelc per call, so every caller that skipped this cache would recompile
// the same package on every construction (a multi-second cost per rec-width
// package). Shared by every CoreML consumer in this backend (AneRecEngine,
// MpsDetector's optional CoreML forward) so the cache — and the failure log —
// exist exactly once.

#ifdef __OBJC__
#import <Foundation/Foundation.h>

namespace turbo_ocr::apple {

// Compiled-model URL for `path` (a .mlpackage), or nil with a logged error.
// Thread-safe; the compile runs at most once per path per process.
NSURL *coreml_compiled_url(NSString *path);

} // namespace turbo_ocr::apple
#endif
