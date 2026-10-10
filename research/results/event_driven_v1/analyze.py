"""Rebuild bounded decision, tables and figure from measured artifacts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
B = json.loads((ROOT / "benchmark.json").read_text())
A = json.loads((ROOT / "audit.json").read_text())
FAST = [r for r in B["measurements"] if r["backend"] == "tiktoken"]


def main():
    target = [r for r in FAST if r["K"] == 65536]
    exact = all(r["exact_ids_all_rounds"] for r in B["measurements"]) and all(
        r["exact"] for r in A["audit"])
    speed_gate = all(r["methods"]["local"]["speedup_over_full_median"] >= 1.1 for r in target)
    no_regression = all(r["methods"]["local"]["speedup_over_full_median"] >= 1 for r in FAST)
    decision = dict(decision="GO" if exact and speed_gate and no_regression else
                    "NO_GO_FOR_CURRENT_PYTHON_LOCAL_IMPLEMENTATION",
                    exact_output_gate=exact, fast_64K_10pct_gate=speed_gate,
                    fast_no_regression_gate=no_regression, source_sha=B["source_sha"],
                    baseline_sha=B["baseline_sha"], audit_conditions=sum(r["documents"] for r in A["audit"]),
                    real_document_model_backend_conditions=sum(r["documents"] for r in B["measurements"]),
                    recommended_team_default="full", research_default="existing event skip",
                    LM_retraining=False, historical_BPB_reused=False, publication_novelty="not established")
    (ROOT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    rows = ["| K | Backend | Domain | Full MB/s | Skip MB/s | Local MB/s | Local/full paired × | Range |",
            "|---:|---|---|---:|---:|---:|---:|---|" ]
    for r in B["measurements"]:
        m = r["methods"]
        ratio = m["local"]["speedup_over_full_median"]
        lo, hi = m["local"]["speedup_range"]
        rows.append(f"| {r['K']} | {r['backend']} | {r['domain']} | "
                    f"{m['full']['MBps_median']:.3f} | {m['skip']['MBps_median']:.3f} | "
                    f"{m['local']['MBps_median']:.3f} | {ratio:.3f} | {lo:.3f}–{hi:.3f} |")
    (ROOT / "TIMINGS.md").write_text("\n".join(rows) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8), layout="constrained")
    x = np.arange(len(FAST))
    for j, (mode, color) in enumerate(zip(("scaffold", "full", "skip", "local"),
                                         ("#c4cad1", "#697386", "#e9a044", "#368f86"))):
        values = [r["methods"][mode]["MBps_median"] for r in FAST]
        bars = axes[0].bar(x + (j - 1.5) * .19, values, width=.19, label=mode, color=color)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    labels = [f"{r['K'] // 1024}K\n{'Wiki' if r['domain'] == 'wiki_dev_sel' else 'PG19'}" for r in FAST]
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Encoding throughput (decimal MB/s)")
    axes[0].set_title("Existing fast backend: no consistent gain")
    axes[0].legend(frameon=False, ncols=4, fontsize=9)
    axes[0].set_ylim(0, 16)
    ratios = np.array([r["methods"]["local"]["speedup_over_full_median"] for r in B["measurements"]])
    bounds = np.array([r["methods"]["local"]["speedup_range"] for r in B["measurements"]])
    y = np.arange(len(ratios))
    names = [f"{r['K'] // 1024}K {r['backend']} {'Wiki' if r['domain'] == 'wiki_dev_sel' else 'PG19'}"
             for r in B["measurements"]]
    axes[1].errorbar(ratios, y, xerr=np.stack((ratios - bounds[:, 0], bounds[:, 1] - ratios)),
                     fmt="o", color="#ac4552", capsize=3)
    axes[1].axvline(1, color="#697386", linestyle="--", linewidth=1)
    axes[1].axvline(1.1, color="#368f86", linestyle=":", linewidth=1)
    axes[1].set_yticks(y, names, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlim(.8, 1.12)
    axes[1].set_xlabel("Paired local / full speedup; min–max over 7 rounds")
    axes[1].set_title("Output parity passes; speed gate fails")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Event-Driven H-Scaffold — Apple M4 Pro, fixed 94 documents", fontsize=14)
    fig.savefig(ROOT / "RESULTS.png", dpi=180)
    fig.savefig(ROOT / "RESULTS.svg")
    print(decision["decision"], "exact conditions:", decision["audit_conditions"])


if __name__ == "__main__":
    main()
