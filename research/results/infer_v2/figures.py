"""Figures for infer_v2 (reads frozen JSON in this directory and ../infer_v1)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
an = json.load(open(f"{D}/analysis_v2.json"))
v1 = json.load(open(f"{D}/../infer_v1/analysis.json"))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

ax = axes[0]
styles = {"scaffold_canonical": ("trained on canonical", "C0", "o"), "scaffold_dp": ("trained on DP", "C3", "s"),
          "scaffold_longest": ("trained on longest", "C2", "^"), "scaffold_mix": ("trained on mix", "C4", "D")}
for arm, (lab, col, mk) in styles.items():
    rows = sorted(an["frontier"][arm].items(), key=lambda x: x[1]["tokens_rel"])
    xs = [100 * r["tokens_rel"] for _, r in rows]
    ys = [r["dbpb_vs_canonical_seg"] for _, r in rows]
    ax.plot(xs, ys, marker=mk, color=col, label=lab, lw=1.2, ms=5)
    if arm == "scaffold_canonical":
        h = an["frontier"][arm]["seg_hscaffold"]
        ax.annotate("H-Scaffold", (100 * h["tokens_rel"], h["dbpb_vs_canonical_seg"]), xytext=(-0.25, 0.03),
                    arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
c = v1["arms"]["scaffold_canonical"]
ax.scatter([-0.570], [c["test_dp"]["mean"] - c["test_canonical"]["mean"]], marker="x", s=70, color="C0",
           label="canonical model, DP longest-first tie-break (E2)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("test tokens vs canonical Scaffold (%)")
ax.set_ylabel("ΔBPB vs same model on canonical segmentation")
ax.set_title("① λ frontier: #tokens + λ·#boundary changes\n(λ = 1 → canonical, λ = 0 → min tokens)")
ax.legend(fontsize=7, loc="center right")

ax = axes[1]
arms = ["scaffold_canonical", "scaffold_auxshuf", "scaffold_aux"]
labels = ["no feature", "shuffled ids\n(control)", "scaffold ids"]
f = an["features"]["arms"]
for i, a in enumerate(arms):
    ps = [v[0] for v in f[a]["per_seed"].values()]
    ax.scatter([i] * len(ps), ps, s=22, color=f"C{i}")
    ax.scatter([i], [f[a]["test_mean"]], marker="_", s=500, color="k")
q = os.path.join(D, "quiche_replication/lm")
for i, a in enumerate(arms[1:], 1):
    for s in (0, 2):
        p = os.path.join(q, a, f"seed{s}", "results.json")
        if os.path.exists(p):
            ax.scatter([i + 0.15], [json.load(open(p))["bpb"]["test"]], s=22, facecolors="none", edgecolors=f"C{i}")
ax.set_xticks(range(3), labels)
ax.set_ylabel("test BPB (wine filled, quiche hollow)")
ax.set_title("③ causal scaffold input features (fire at 0.25% of positions)")

fig.tight_layout()
fig.savefig(f"{D}/RESULTS.png", dpi=150)
print("wrote RESULTS.png")
