// Dev/CI driver for the in-process formula FAST backends. Constructs PPFormulaNetOrt
// directly and runs its public gate_bench() over the 30 gate crops — emits
// /tmp/cpp_<backend>_{tokens,latex}.json (+ _cont_tokens.json for plus-M) and prints
// lockstep crops/s. No bench code lives in the server's load/run path.
//
//   ./plusm_selftest [backend] [model_dir] [tokenizer_json] [gate_dir]
#include <cstdlib>
#include <iostream>
#include <string>

#include "nvidia/stages/ppformulanet_ort.h"

int main(int argc, char **argv) {
  std::string backend = argc > 1 ? argv[1] : "ppformulanet_plus_m";
  std::string dir = argc > 2 ? argv[2] : ("models/formula/" + backend);
  std::string tok = argc > 3 ? argv[3] : (dir + "/tokenizer.json");
  std::string gate = argc > 4 ? argv[4] : "models/formula/ppformulanet_plus_m/eval";

  turbo_ocr::formula::PPFormulaNetOrt rec(backend);
  std::cerr << "[selftest] backend=" << backend << " load_model_dir: " << dir << '\n';
  if (!rec.load_model_dir(dir)) { std::cerr << "[selftest] load_model_dir FAILED\n"; return 3; }
  std::cerr << "[selftest] load_tokenizer: " << tok << '\n';
  if (!rec.load_tokenizer(tok)) { std::cerr << "[selftest] load_tokenizer FAILED\n"; return 4; }
  std::cerr << "[selftest] ready=" << rec.is_ready() << " backend=" << rec.backend_name() << '\n';
  if (!rec.gate_bench(gate)) { std::cerr << "[selftest] gate_bench FAILED\n"; return 5; }
  return 0;
}
