"""Pure helpers for the release speed test: parse, median, change, table."""
import json
import statistics

MARKER = "SPEEDTEST "
_KEYS = ("total", "chemistry", "transport")
_COLUMNS = "| Model | Cells | Steps | Version | Total (s) | Chemistry (s) | Transport (s) | Other (s) | Change |"


def parse_result(stdout):
    """Return the JSON dict on the last ``SPEEDTEST`` line of ``stdout``, or None."""
    lines = [ln for ln in stdout.splitlines() if ln.startswith(MARKER)]
    if not lines:
        return None
    return json.loads(lines[-1][len(MARKER):])


def median_result(results):
    """Per-key median over the repeats that finished; None if none did."""
    done = [r for r in results if r is not None]
    if not done:
        return None
    out = {"cells": done[0].get("cells"), "steps": done[0].get("steps")}
    for key in _KEYS:
        values = [r.get(key) for r in done]
        out[key] = None if any(v is None for v in values) else statistics.median(values)
    return out


def pct_change(old, new):
    if old is None or new is None or old == 0:
        return None
    return (new - old) / old * 100.0


def _other(r):
    if r is None or None in (r["total"], r["chemistry"], r["transport"]):
        return None
    return r["total"] - r["chemistry"] - r["transport"]


def _pair(old, new, get):
    def fmt(r):
        if r is None:
            return "failed"
        v = get(r)
        return "n/a" if v is None else f"{v:.1f}"
    return f"{fmt(old)} / {fmt(new)}"


def format_table(rows, old_version, new_version, header):
    """Markdown table, one row per model, old and new values side by side."""
    lines = [header, "", _COLUMNS, "|" + "---|" * 9]
    for model, old, new in rows:
        ref = new or old or {}
        change = pct_change(old and old["total"], new and new["total"])
        lines.append(
            f"| {model} | {ref.get('cells', 'n/a')} | {ref.get('steps', 'n/a')} "
            f"| {old_version} / {new_version} "
            f"| {_pair(old, new, lambda r: r['total'])} "
            f"| {_pair(old, new, lambda r: r['chemistry'])} "
            f"| {_pair(old, new, lambda r: r['transport'])} "
            f"| {_pair(old, new, _other)} "
            f"| {'n/a' if change is None else f'{change:+.0f}%'} |"
        )
    return "\n".join(lines) + "\n"
