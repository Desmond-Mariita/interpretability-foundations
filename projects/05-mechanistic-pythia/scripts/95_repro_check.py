"""Verify the re-run's metrics.json reproduces the v1.0 REPORT.md tables (reconstruction gate).

One-off verification script (not part of the frozen pipeline): compares every
(property, point) balanced accuracy / control / selectivity in outputs/metrics.json
against the values recorded in REPORT.md within a documented tolerance, and prints a
per-point diff table. Used to satisfy ADR 006 stop condition 3 (faithful reconstruction).
"""

from __future__ import annotations

import json
import re

from _paths import OUTPUTS

TOL = 0.004  # float extraction nondeterminism across machines (documented)


def parse_report_table(text: str) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Parse REPORT.md per-property tables: {prop: {point: (ba, ctrl_ba, selectivity)}}."""
    sections = re.split(r"^### 4\.\d ", text, flags=re.M)
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for sec in sections[1:]:
        head = sec.split("\n", 1)[0].strip()
        prop = head.split("--")[0].strip().strip("`")
        rows = {}
        for m in re.finditer(
            r"^\| (\w+) \| ([\d.]+) \| \[[^\]]*\] \| ([\d.]+) \| (-?[\d.]+) \|", sec, flags=re.M
        ):
            rows[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        if rows:
            out[prop] = rows
    return out


def main() -> None:  # pragma: no cover - verification tooling
    from pathlib import Path

    report = Path("REPORT.md").read_text(encoding="utf-8")
    recorded = parse_report_table(report)
    metrics = json.loads((OUTPUTS / "metrics.json").read_text())

    all_ok = True
    for prop, points in metrics["properties"].items():
        for pt in points["points"]:
            name = pt["point"]
            if name not in recorded.get(prop, {}):
                continue
            r_ba, r_ctrl, r_sel = recorded[prop][name]
            d_ba = abs(pt["balanced_acc"] - r_ba)
            d_sel = abs(pt["selectivity"] - r_sel)
            d_ctrl = abs(pt["control_balanced_acc"] - r_ctrl)
            ok = max(d_ba, d_ctrl, d_sel) <= TOL
            all_ok &= ok
            flag = "OK " if ok else "DIFF"
            print(
                f"{flag} {prop:<12} {name:<10} ba {pt['balanced_acc']:.4f} vs {r_ba:.4f}"
                f" (d={d_ba:.4f})  ctrl {pt['control_balanced_acc']:.4f} vs {r_ctrl:.4f}"
                f" (d={d_ctrl:.4f})  sel {pt['selectivity']:.4f} vs {r_sel:.4f} (d={d_sel:.4f})"
            )
    print(f"tolerance: {TOL} (float extraction nondeterminism across machines)")
    print("RECONSTRUCTION:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":  # pragma: no cover
    main()
