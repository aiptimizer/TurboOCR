// Verifies the formula backend swap: for each FORMULA_BACKEND value, resolve the
// routing table the server would build at boot and report the engine the formula
// route lands on. Confirms ppformulanet_plus_m is now a first-class, swappable
// choice alongside the default ppformulanet_s.
#include <cstdlib>
#include <iostream>

#include "turbo_ocr/backend/routing_config.h"

int main() {
  using namespace turbo_ocr::routing;
  for (const char *be : {"ppformulanet_s", "ppformulanet_plus_m", "vlm", "bogus_engine"}) {
    ::setenv("FORMULA_BACKEND", be, 1);
    try {
      RoutingTable t = load_routing_config();
      const BackendSpec *s = resolve(t, "formula");
      std::cout << "FORMULA_BACKEND=" << be << "  ->  engine='"
                << (s ? s->engine : std::string("(null)")) << "'  "
                << (s ? "RESOLVED" : "unrouted") << '\n';
    } catch (const std::exception &e) {
      std::cout << "FORMULA_BACKEND=" << be << "  ->  REJECTED: " << e.what() << '\n';
    }
  }
  return 0;
}
