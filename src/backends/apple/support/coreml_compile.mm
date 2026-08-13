// See coreml_compile.h.

#import "apple/support/coreml_compile.h"

#import <CoreML/CoreML.h>

#include <map>
#include <mutex>
#include <string>

namespace turbo_ocr::apple {

NSURL *coreml_compiled_url(NSString *path) {
  static std::mutex mu;
  static std::map<std::string, NSURL *> cache;
  std::lock_guard<std::mutex> lk(mu);
  const std::string key = path.UTF8String;
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  NSError *err = nil;
  NSURL *c = [MLModel compileModelAtURL:[NSURL fileURLWithPath:path] error:&err];
  if (!c) {
    NSLog(@"[apple] CoreML compile failed for %@: %@", path,
          err.localizedDescription);
    return nil;
  }
  cache[key] = c;
  return c;
}

} // namespace turbo_ocr::apple
