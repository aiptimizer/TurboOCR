# End-to-end drivers

Not a pytest suite and **not** part of `python tests/run_all.py`. Each of
these boots a real server (or a real Docker container), drives it over
HTTP, scores the output and tears it down. They are invoked directly, from
the repo root, and each exits non-zero when a cell fails.

| Driver | Boots | What it proves |
|---|---|---|
| `docker_endpoint_matrix.py` | containers (gpu + cpu images) | every endpoint × every `OCR_LANG` answers, with char-recall scoring. **CI runs this one** against a natively built server via `--base-url`. |
| `docker_language_matrix.py` | containers | the first-boot path a real deployment takes: entrypoint validates `OCR_LANG`, downloads the bundle, loads it. |
| `language_smoketest.py` | `build/turboocr-server` | every advertised language bundle is wired, and catches upstream model regressions. |
| `bench_all_languages.py` | containers | per-language throughput + latency + accuracy table. |

Prerequisites differ per driver: the two `docker_*` ones need
`turbo-ocr:prod` / `turbo-ocr-cpu:prod` built; `language_smoketest.py`
needs a native build; all of them need the per-script fonts listed in
their `CASES` tables (missing fonts are reported as SKIP, not failure).

```bash
python tests/e2e/docker_endpoint_matrix.py --image gpu --only latin
python tests/e2e/language_smoketest.py --only greek,thai
```
