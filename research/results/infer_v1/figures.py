"""Figures for infer_v1 (reads only the frozen JSON artifacts in this directory)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
e01 = json.load(open(f"{D}/e0_e1.json"))
e3 = json.load(open(f"{D}/e3.json"))
an = json.load(open(f"{D}/analysis.json")) if os.path.exists(f"{D}/analysis.json") else None

fig, axes = plt.subplots(1, 3 if an else 2, figsize=(16 if an else 11, 4.2))

# (a) E1: token count relative to Scaffold canonical
ax = axes[0]
methods = ["bpe", "hscaffold", "longest", "dp"]
labels = ["BPE (own vocab)", "H-Scaffold", "Longest-match", "Min-token DP"]
x = range(len(methods))
for i, (K, split) in enumerate([("32768", "test"), ("65536", "test")]):
    r = e01["e1"][K][split]["rel_vs_scaffold"]
    ax.bar([j + 0.38 * i for j in x], [100 * r[m] for m in methods], 0.38, label=f"{int(K)//1024}K")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks([j + 0.19 for j in x], labels, fontsize=8)
ax.set_ylabel("tokens vs Scaffold (%)")
ax.set_title("E1  WikiText-103 test: same Scaffold vocab,\ndifferent inference")
ax.legend(fontsize=8)

# (b) E3: per-language gaps
ax = axes[1]
langs = ["en", "de", "ru", "hi"]
pl = e3["a1.0"]["per_lang"]
g1 = [100 * pl[f"test_{l}"]["bpe_to_scaffold_rel"] for l in langs]
g2 = [100 * pl[f"test_{l}"]["scaffold_dp_gap_rel"] for l in langs]
ax.bar([i - 0.2 for i in range(4)], g1, 0.4, label="BPE → Scaffold (vocab)")
ax.bar([i + 0.2 for i in range(4)], g2, 0.4, label="Scaffold → DP (inference)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(4), langs)
ax.set_ylabel("token count change (%)")
ax.set_title("E3  multilingual 32K (α=1.0), test")
ax.legend(fontsize=8)

if an:
    ax = axes[2]
    arms = [a for a in ("bpe_canonical", "scaffold_canonical", "scaffold_longest", "scaffold_dp", "scaffold_mix")
            if a in an["arms"]]
    for a in arms:
        ps = an["arms"][a]["test"]["per_seed"]
        ax.scatter([a.replace("scaffold_", "S-").replace("_canonical", "")] * len(ps), list(ps.values()), s=18)
        ax.scatter([a.replace("scaffold_", "S-").replace("_canonical", "")], [an["arms"][a]["test"]["mean"]],
                   marker="_", s=400, color="k")
    ax.set_ylabel("test BPB (own segmentation)")
    ax.set_title("E2  42.5M LM, 2 epochs WikiText-103")
    ax.tick_params(axis="x", labelsize=8)

fig.tight_layout()
fig.savefig(f"{D}/RESULTS.png", dpi=150)
print("wrote RESULTS.png")
