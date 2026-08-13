# Decode fuzzing

The server accepts arbitrary PNG / JPEG / PDF bytes from the network. These
targets throw malformed input at the three decode entry points on purpose. It
matters most on the **macOS and Windows** builds, where rendering runs
in-process — a decoder that faults there takes the whole server down. (On Linux
the page renderer is a separate `fastpdf2png` daemon, so a PDFium fault is
contained to a subprocess.)

Run them (Linux + clang):

```bash
sudo apt-get install clang libclang-rt-<ver>-dev   # the -dev package has the
                                                    # libFuzzer/ASan runtime
bash tests/fuzz/run.sh 240                          # seconds per target
```

| target | entry point | first-party? |
|---|---|---|
| `fuzz_ppm_header` | `parse_ppm_header` | **yes** — our parser. The target also asserts the invariants `decode_ppm` relies on (offset within buffer, `payload_bytes == w*h*channels`), so a header that parses "valid" but lies about its geometry is a crash, not a silent bad slice. Runs under ASan. |
| `fuzz_image_decode` | `decode_cpu_fallback` | **mixed** — our Wuffs PNG wrapper + `INT_MAX` guard (first-party), then `cv::imdecode` (OpenCV). Exactly what `/ocr/raw` runs on the body. ASan. |
| `fuzz_pdf_open` | `FPDF_LoadMemDocument` + first-page load | third-party (PDFium). A regression net for the in-process crash surface; fuzzer-only because the vendored PDFium is not ASan-instrumented. |

A crash writes a `crash-<hash>` reproducer. Triage: a first-party frame is a bug
to fix here; a pure OpenCV/PDFium fault is an upstream finding, and the fix we
own is usually a bound applied *before* the hand-off (the PPM area-bomb guard
and the `INT_MAX` size check are exactly that kind of guard).
