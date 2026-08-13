"""Regression detector: rolling baseline + verdict for cua-bench output.

State on disk:
  /tmp/cua_bench/
    <ts>.json          per-run report
    latest.json        -> symlink to most recent <ts>.json
    latest.summary     human one-liner
    baseline.json      rolling state: per-scenario list of recent PASS p50s
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASELINE_WINDOW = 3  # median of last 3 PASS p50s


# -- Baseline I/O ----------------------------------------------------------------

def load_baseline(out_dir: Path) -> dict[str, Any]:
    p = out_dir / "baseline.json"
    if not p.exists():
        return {"scenarios": {}, "consecutive_alerts": {}}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"scenarios": {}, "consecutive_alerts": {}}


def save_baseline(out_dir: Path, baseline: dict[str, Any]) -> None:
    (out_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))


def baseline_p50(baseline: dict[str, Any], scenario: str) -> float | None:
    entries = baseline.get("scenarios", {}).get(scenario, [])
    samples = [e["p50_ms"] for e in entries[-BASELINE_WINDOW:]]
    if not samples:
        return None
    return statistics.median(samples)


def baseline_delta(baseline: dict[str, Any], scenario: str) -> float | None:
    """Prior (form_p50 - text_p50) tracked for formula/table scenarios."""
    entries = baseline.get("scenarios", {}).get(scenario, [])
    deltas = [e["delta_ms"] for e in entries[-BASELINE_WINDOW:] if "delta_ms" in e]
    if not deltas:
        return None
    return statistics.median(deltas)


def append_pass(baseline: dict[str, Any], scenario: str, p50_ms: float,
                ts: str, delta_ms: float | None = None) -> None:
    s = baseline.setdefault("scenarios", {}).setdefault(scenario, [])
    entry = {"timestamp": ts, "p50_ms": p50_ms}
    if delta_ms is not None:
        entry["delta_ms"] = delta_ms
    s.append(entry)
    # keep the last (WINDOW * 2) so we can recover from a single bad sample
    del s[: max(0, len(s) - BASELINE_WINDOW * 2)]


def reset_consecutive(baseline: dict[str, Any], scenario: str) -> None:
    baseline.setdefault("consecutive_alerts", {})[scenario] = 0


def bump_consecutive(baseline: dict[str, Any], scenario: str) -> int:
    c = baseline.setdefault("consecutive_alerts", {})
    c[scenario] = int(c.get(scenario, 0)) + 1
    return c[scenario]


# -- Verdict logic ---------------------------------------------------------------

@dataclass
class ScenarioResult:
    name: str
    p50_ms: float | None      # None if scenario failed to produce numbers
    p99_ms: float | None
    err_rate: float
    delta_ms: float | None = None  # vs text_p50 (formula/table)


def compute_verdicts(
    results: list[ScenarioResult],
    baseline: dict[str, Any],
    text_p50_target: float = 270.0,
    text_p50_hard: float = 280.0,
    text_p50_halt: float = 300.0,
) -> dict[str, Any]:
    """Returns the `verdict` block for the run JSON. Mutates `baseline` to
    record PASS samples and increment/reset consecutive_alerts counters.
    """
    by_name = {r.name: r for r in results}
    text = by_name.get("text_only")
    reasons: list[str] = []
    per_scn: dict[str, str] = {}

    # ---- text_only ----
    text_verdict = "PASS"
    if text is None or text.p50_ms is None:
        text_verdict = "INFRA"
        reasons.append("text_only: no measurements")
    else:
        if text.err_rate > 0.10:
            text_verdict = "INFRA"
            reasons.append(f"text_only: err_rate {text.err_rate:.2%} > 10%")
        elif text.p50_ms > text_p50_halt:
            text_verdict = "HALT"
            reasons.append(f"text_only: p50 {text.p50_ms:.1f}ms > halt ceiling {text_p50_halt}ms")
        else:
            B = baseline_p50(baseline, "text_only")
            if B is None:
                text_verdict = "PASS"
                reasons.append("text_only: no baseline yet")
            elif text.p50_ms > 1.10 * B:
                text_verdict = "HALT"
                reasons.append(f"text_only: p50 {text.p50_ms:.1f}ms > 1.10 * B ({B:.1f})")
            elif text.p50_ms > text_p50_hard:
                text_verdict = "ALERT"
                reasons.append(f"text_only: p50 {text.p50_ms:.1f}ms > hard ceiling {text_p50_hard}ms")
            elif text.p50_ms > 1.05 * B:
                text_verdict = "ALERT"
                reasons.append(f"text_only: p50 {text.p50_ms:.1f}ms > 1.05 * B ({B:.1f})")

    per_scn["text_only"] = text_verdict

    # ---- formula_heavy / table_heavy: delta-vs-text comparison ----
    text_p50 = text.p50_ms if (text and text.p50_ms is not None) else None
    for name in ("formula_heavy", "table_heavy"):
        r = by_name.get(name)
        if r is None or r.p50_ms is None or text_p50 is None:
            per_scn[name] = "INFRA"
            reasons.append(f"{name}: no measurements")
            continue
        if r.err_rate > 0.10:
            per_scn[name] = "INFRA"
            reasons.append(f"{name}: err_rate {r.err_rate:.2%} > 10%")
            continue
        delta = r.p50_ms - text_p50
        r.delta_ms = delta
        prior = baseline_delta(baseline, name)
        if prior is not None and prior > 0 and delta > 1.20 * prior:
            per_scn[name] = "ALERT"
            reasons.append(f"{name}: Δ {delta:.1f}ms > 1.20 * prior Δ ({prior:.1f}ms)")
        else:
            per_scn[name] = "PASS"

    # ---- consecutive ALERT → HALT for text_only relative drift only ----
    consecutive: dict[str, int] = {}
    if text_verdict == "PASS":
        reset_consecutive(baseline, "text_only")
    elif text_verdict == "ALERT" and text and text.p50_ms is not None and text.p50_ms <= text_p50_hard:
        # only the +5% rule counts toward "two consecutive ALERT → HALT"
        n = bump_consecutive(baseline, "text_only")
        if n >= 2:
            text_verdict = "HALT"
            per_scn["text_only"] = "HALT"
            reasons.append("text_only: two consecutive ALERTs → HALT")
    # else: hard-ceiling ALERT keeps the counter where it is, no auto-promote
    consecutive["text_only"] = baseline.get("consecutive_alerts", {}).get("text_only", 0)

    # ---- overall ----
    severities = ["PASS", "ALERT", "INFRA", "HALT"]
    overall = max(per_scn.values(), key=severities.index)
    return {
        "overall": overall,
        "text_only": per_scn["text_only"],
        "formula_heavy": per_scn.get("formula_heavy", "INFRA"),
        "table_heavy": per_scn.get("table_heavy", "INFRA"),
        "consecutive_alerts": consecutive,
        "reasons": reasons,
    }


# -- Output writers --------------------------------------------------------------

def write_report(out_path: Path, report: dict[str, Any]) -> None:
    out_path.write_text(json.dumps(report, indent=2, default=str))


def update_latest_symlink(out_dir: Path, target: Path) -> None:
    link = out_dir / "latest.json"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
    except OSError:
        pass
    try:
        link.symlink_to(target.name)
    except OSError:
        # fallback: write contents
        link.write_text(target.read_text())


def write_summary_line(out_dir: Path, line: str) -> None:
    (out_dir / "latest.summary").write_text(line + "\n")


# -- Exit-code mapping -----------------------------------------------------------

EXIT_PASS = 0
EXIT_INFRA = 1
EXIT_SERVER_DOWN = 2
EXIT_ALERT = 3
EXIT_HALT = 4


def exit_code_for(verdict: str) -> int:
    return {
        "PASS": EXIT_PASS,
        "INFRA": EXIT_INFRA,
        "ALERT": EXIT_ALERT,
        "HALT": EXIT_HALT,
    }.get(verdict, EXIT_INFRA)
