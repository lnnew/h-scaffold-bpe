"""Produce the fixed-trial decision and figure; do not tune dispatch thresholds."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    bench = json.loads((ROOT / "benchmark.json").read_text())
    short = json.loads((ROOT / "short_amortized.json").read_text())
    audit = json.loads((ROOT / "audit.json").read_text())
    long = [r for r in bench["measurements"] if r["domain"] != "short_pretokens"]
    rows = long + short["measurements"]
    exact = all(r["exact_ids_all_rounds"] for r in rows) and all(r["exact"] for r in audit["audit"])
    long_gate = all(r["methods"]["hybrid"]["speedup_over_full_median"] >= 1.05
                    and r["methods"]["hybrid"]["speedup_range"][0] > 1 for r in long)
    short_gate = all(r["methods"]["hybrid"]["speedup_over_full_median"] >= 1
                     and r["methods"]["hybrid"]["speedup_range"][0] >= .98
                     for r in short["measurements"])
    decision = dict(decision="GO" if exact and long_gate and short_gate else "NO_GO_STOP_LOCAL_OPTIMIZATION",
                    correctness=exact, long_speed_gate=long_gate, short_non_regression_gate=short_gate,
                    source_sha=bench["source_sha"], baseline_sha=bench["baseline_sha"],
                    dispatch=bench["dispatch"], thresholds_retuned=False, additional_optimization=False,
                    hybrid_differential_conditions=sum(r["documents"] for r in audit["audit"]
                                                       if "hybrid" in r["compared_methods"]),
                    LM_training=False, utility_design=False, team_default="full")
    (ROOT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    table = ["| K | Backend | Domain | Full MB/s | Local MB/s | Hybrid MB/s | Paired Hybrid/full × | Min–max |",
             "|---:|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        m = r["methods"]
        h = m["hybrid"]
        lo, hi = h["speedup_range"]
        table.append(f"| {r['K']} | {r['backend']} | {r['domain']} | {m['full']['MBps_median']:.3f} | "
                     f"{m['local']['MBps_median']:.3f} | {h['MBps_median']:.3f} | "
                     f"{h['speedup_over_full_median']:.3f} | {lo:.3f}–{hi:.3f} |")
    (ROOT / "TIMINGS.md").write_text("\n".join(table) + "\n")
    native = [r for r in rows if r["backend"] == "tiktoken"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8), layout="constrained")
    x = np.arange(len(native))
    for j, (name, color) in enumerate(zip(("full", "local", "hybrid"), ("#677384", "#cfab65", "#ab4757"))):
        bars = axes[0].bar(x + (j - 1) * .23, [r["methods"][name]["MBps_median"] for r in native],
                           width=.23, label=name, color=color)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=7, padding=3)
    domain = {"wiki_dev_sel": "Wiki", "pg19_excerpts": "PG19", "short_pretokens": "Short"}
    axes[0].set_xticks(x, [f"{r['K'] // 1024}K\n{domain[r['domain']]}" for r in native])
    axes[0].set_ylim(0, 11)
    axes[0].set_ylabel("Encoding throughput (decimal MB/s)")
    axes[0].set_title("Frozen original vs one fixed Hybrid")
    axes[0].legend(frameon=False, ncols=3)
    ratios = np.array([r["methods"]["hybrid"]["speedup_over_full_median"] for r in rows])
    bounds = np.array([r["methods"]["hybrid"]["speedup_range"] for r in rows])
    y = np.arange(len(rows))
    axes[1].errorbar(ratios, y, xerr=np.stack((ratios - bounds[:, 0], bounds[:, 1] - ratios)),
                     fmt="o", color="#ab4757", capsize=3)
    axes[1].axvline(1, color="#677384", linestyle="--", linewidth=1)
    axes[1].set_yticks(y, [f"{r['K'] // 1024}K {r['backend']} {domain[r['domain']]}" for r in rows], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlim(.7, 1.2)
    axes[1].set_xlabel("Paired Hybrid / full speedup; 7-round min–max")
    axes[1].set_title("Correctness passes; acceleration fails")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Hybrid H-Scaffold — Apple M4 Pro; short calls amortized ×128", fontsize=14)
    fig.savefig(ROOT / "RESULTS.png", dpi=180)
    fig.savefig(ROOT / "RESULTS.svg")
    svg = ROOT / "RESULTS.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()).rstrip() + "\n")
    print(decision["decision"], "exact Hybrid audit conditions:", decision["hybrid_differential_conditions"])


if __name__ == "__main__":
    main()
