"""Recompute paired results from preserved document losses and export a figure."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

root = Path(__file__).resolve().parent
contract = json.loads((root / "contract.json").read_text())
decision = json.loads((root / "decision.json").read_text())
methods, seeds = contract["methods"], contract["seeds"]
results = {(m, s): json.loads((root / f"runs/{m}/seed{s}/results.json").read_text())
           for m in methods for s in seeds}
assert len(results) == 12
for key in ["raw_train_hash", "train_order_hash", "train_exposure_bytes", "steps", "n_params"]:
    assert len({r[key] for r in results.values()}) == 1, key
assert all(r["skipped_updates"] == 0 and r["steps"] == contract["steps"] for r in results.values())
comparisons, max_error = {}, 0.
for split in ["pg19", "wiki", "val"]:
    comparisons[split] = {}
    for m in methods:
        mean = np.mean([results[m, s]["bpb"][split] for s in seeds])
        max_error = max(max_error, abs(mean - decision["means"][split][m]))
        for s in seeds:
            a = np.load(root / f"runs/{m}/seed{s}/eval_{split}.npz")
            observed = a["doc_nll_bits"].sum() / a["doc_bytes"].sum()
            assert abs(observed - results[m, s]["bpb"][split]) < 1e-12
    for ref in ["frequency", "word", "random"]:
        rows, deltas = [], []
        for s in seeds:
            a = np.load(root / f"runs/future/seed{s}/eval_{split}.npz")
            b = np.load(root / f"runs/{ref}/seed{s}/eval_{split}.npz")
            sizes = a["doc_bytes"]
            assert np.array_equal(sizes, b["doc_bytes"])
            d = a["doc_nll_bits"] - b["doc_nll_bits"]
            rows.append(d); deltas.append(float(d.sum() / sizes.sum()))
        d = np.mean(rows, axis=0)
        idx = np.random.default_rng(0).integers(len(sizes), size=(4000, len(sizes)))
        interval = np.quantile(d[idx].sum(1) / sizes[idx].sum(1), [.025, .975])
        row = dict(delta_bpb=float(d.sum() / sizes.sum()), delta_by_seed=deltas,
                   doc_ci95=interval.tolist(), all_seeds_improve=all(v < 0 for v in deltas))
        expected = decision["comparisons"][split][ref]
        for key in ["delta_bpb", "delta_by_seed", "doc_ci95"]:
            max_error = max(max_error, float(np.max(np.abs(np.asarray(row[key]) - expected[key]))))
        assert row["all_seeds_improve"] == expected["all_seeds_improve"]
        comparisons[split][ref] = row
assert max_error < 1e-12
gate = all(v["all_seeds_improve"] and v["doc_ci95"][1] < 0 for v in comparisons["pg19"].values())
assert gate == (decision["status"] == "GO_TO_INDEPENDENT_REPLICATION")
audit = dict(status="PASS", fits=12, source_sha=contract["source_sha"], max_stat_error=max_error,
             raw_exposure_bytes=results["frequency", 0]["train_exposure_bytes"],
             skipped_updates=0, comparisons=comparisons)
(root / "AUDIT.json").write_text(json.dumps(audit, indent=2))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
labels = {"frequency": "Frequency", "word": "Word utility", "future": "Future utility", "random": "Matched random"}
colors = {"frequency": "#68788c", "word": "#bf8522", "future": "#207b74", "random": "#8460a4"}
for i, m in enumerate(methods):
    values = [results[m, s]["bpb"]["pg19"] for s in seeds]
    axes[0].scatter(np.full(len(values), i) + np.linspace(-.07, .07, len(values)), values,
                    color=colors[m], s=35, alpha=.8)
    axes[0].plot([i - .22, i + .22], [np.mean(values)] * 2, color=colors[m], lw=3)
axes[0].set(xticks=range(len(methods)), xticklabels=[labels[m].replace(" ", "\n") for m in methods],
            ylabel="PG-19 bits per byte (lower is better)", title="Three independent initialization seeds")
axes[0].grid(axis="y", alpha=.2)
axes[0].ticklabel_format(useOffset=False, axis="y")
for i, ref in enumerate(["frequency", "word", "random"]):
    row = comparisons["pg19"][ref]; lo, hi = row["doc_ci95"]
    axes[1].plot([lo, hi], [i, i], color=colors[ref], lw=3)
    axes[1].scatter(row["delta_bpb"], i, color=colors[ref], s=40, zorder=3)
axes[1].axvline(0, color="#777", ls="--", lw=1)
axes[1].set(yticks=range(3), yticklabels=["Future − " + labels[r] for r in ["frequency", "word", "random"]],
            xlabel="Δ BPB; negative favors Future", title="95% document bootstrap intervals")
axes[1].invert_yaxis(); axes[1].grid(axis="x", alpha=.2)
axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
fig.suptitle("Fixed K=32,768 · 32 slot exchanges · 42.51M LM · 1,500 updates", fontsize=12)
for extension in ["png", "svg"]:
    path = root / f"pg19_comparison.{extension}"
    fig.savefig(path, dpi=240)
    if extension == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
print(json.dumps(dict(status="PASS", fits=12, max_error=max_error, decision=decision["status"], means=decision["means"])))
