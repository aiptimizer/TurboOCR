# Pre-commit checks

Install once per clone:

```bash
bash scripts/setup/install_hooks.sh
```

The hook (`scripts/git-hooks/pre-commit`) runs static checks on **staged
files only** and is hard-capped at **20 seconds** — cheap greps first, then a
time-boxed cppcheck with whatever budget remains.

Each check maps to a defect class that has actually shipped in this repo:

| Check | Why it exists |
|---|---|
| No `std::cout` / `std::cerr` in `src/` | Diagnostics bypassed the structured logger's level control and per-site rate limiting — request-path prints were a log-flood surface. |
| No raw `getenv` in `src/`, `include/` | Config knobs were parsed by hand-pasted `to_int` copies with drifting semantics; `turbo_ocr::env` (common/env_utils.h) is the one parser. |
| No `std::stoi` family | An unguarded `std::stoi` on a daemon reply threw `std::invalid_argument` past the error taxonomy. Use `std::from_chars` with explicit handling. |
| No `file(GLOB)` in CMake | Explicit source lists only. |
| No orphaned `CMakeLists.txt` | Four dead ones misled contributors about the build layout. |
| cppcheck (warning/performance/portability) | General static net, staged `.cpp` only, time-boxed. |

Deliberate exceptions are annotated in-line
(`// pre-commit-allow-stream`, `// pre-commit-allow-getenv`,
`// pre-commit-allow-stoi`) or, for a one-off commit, `git commit --no-verify`
with the reason in the commit message.

## File-length ratchet (max 500 lines)

Any staged `.cpp/.h/.cu/.cuh` over 500 lines fails the commit. The limit is a
ratchet: pre-existing oversized files trip it the first time they are touched,
which is when they get split. Escape hatch for the rare genuinely-unsplittable
file: `// pre-commit-allow-length` within the first 5 lines, next to a short
justification.
