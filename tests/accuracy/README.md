# Accuracy suite

Ground-truth scoring for every image/PDF fixture across every endpoint.

## Important: GT is captured-from-OCR, not hand-labelled

The `expected/*.json` files under `tests/fixtures/images/expected/` and
`tests/fixtures/pdf/expected/` were produced by an earlier run of this
same OCR pipeline. They represent *baseline output*, not absolute truth.

What this suite actually measures: **regression against the captured
baseline**. An F1 drop means today's model differs from the day the
baseline was captured — possibly for the worse. Use
`tests/accuracy/floors.json` to set per-fixture floors conservative
enough that healthy noise doesn't trip CI.

## Running

```
python tests/run_all.py --suite accuracy
# or
pytest tests/accuracy -v
```

Layout is on by default; the layout suite additionally needs
`ENABLE_LAYOUT=1` **in the test environment** (the pytest gate in
`test_layout_accuracy.py`), not on the server — setting `ENABLE_LAYOUT` on
the server is a **fatal boot error** (the variable was retired; layout is
always on and `LAYOUT_DISABLED=1` is the off switch).

## Re-capturing the baseline

If you intentionally change the model and want to reset the floor, run
the suite with floors set to 0.0 to capture current scores, then bump
`floors.json` to `measured - 0.05`.
