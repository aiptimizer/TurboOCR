# Vendor SDK stubs for cross-vendor type checking

TurboOCR compiles a different set of sources per vendor: `TURBO_BACKENDS=cpu;apple`
on a Mac never compiles `src/backends/nvidia/**`, `src/backends/amd/**`,
`src/backends/intel/**`, or any `*_gpu.cpp` route. That is fine for shipping —
each arm is built on its own hardware — but it means a change to a SHARED
signature (a route registrar, `StageAvailability`, `validate_request`) can be
edited in those files and never once seen by a compiler until someone with a
CUDA box builds it.

These headers close that gap. They declare just enough of the CUDA runtime,
TensorRT, nvJPEG, HIP and MIGraphX APIs for `clang -fsyntax-only` to type-check
every vendor source on any machine.

## Running

    tools/syntax_shims/check.sh                  # everything in sources.txt
    tools/syntax_shims/check.sh path/to/one.cpp  # a single file

Exit status is non-zero if any file fails, so it works as a pre-merge gate.

## What this does and does not prove

Proves: declarations, types, overload resolution, and template instantiation all
line up across the vendor arms — i.e. the code would *compile*.

Does NOT prove: that it runs, links, or produces correct results. The stubs have
no bodies and are never linked into anything. Behaviour on NVIDIA/AMD/Intel
still requires a build and a test run on that hardware.

Treat a green run as "the refactor did not break the build", not "the backend
works".

## Keeping the stubs honest

A stub only needs the API surface the tree actually touches, so it grows when
vendor code starts calling something new — the failure mode is a loud
`error: use of undeclared identifier`, which is exactly the signal to add the
declaration. Copy the real signature from the vendor SDK; a stub that *disagrees*
with the real header is worse than no stub, because it would type-check code
that cannot compile on the target.

These files are never on the include path of a real build (nothing in
`CMakeLists.txt` references this directory) — only `check.sh` adds them.
