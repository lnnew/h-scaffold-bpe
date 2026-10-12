"""Figure for the Scaffold strength sweep (reads only frozen JSON in this directory)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
an = json.load(open(f"{D}/analysis_strength.json"))
intr = json.load(open(f"{D}/intrinsic.json"))
cs = sorted(an["arms"], key=float)
x = list(range(len(cs)))
lab = ["0\n(BPE)" if c == "0" else ("1\n(Scaffold)" if c == "1" else c) for c in cs]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

ax = axes[0]
for i, c in enumerate(cs):
    ps = list(an["arms"][c]["per_seed"].values())
    ax.scatter([i] * len(ps), ps, s=16, color="C0", alpha=0.6)
    ax.scatter([i], [an["arms"][c]["test_mean"]], marker="_", s=500, color="k")
ax.set_xticks(x, lab)
ax.set_xlabel("scaffold factor c  (scaffold iff f(x) < c·f(Qhead))")
ax.set_ylabel("test BPB (dots = seeds, bar = mean)")
ax.set_title("A  LM quality vs scaffold strength (42.5M, 2 epochs)")

ax = axes[1]
base = an["arms"]["0"]["test_tokens"]
ax.plot(x, [100 * (an["arms"][c]["test_tokens"] / base - 1) for c in cs], marker="o", color="C1", label="test tokens vs BPE (%)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x, lab)
ax.set_ylabel("token count change vs BPE (%)", color="C1")
ax2 = ax.twinx()
ax2.plot(x, [an["arms"][c]["test_mean"] - an["arms"]["0"]["test_mean"] for c in cs], marker="s", color="C0")
ax2.set_ylabel("ΔBPB vs BPE", color="C0")
ax.set_title("compression gets worse while BPB gets better")

ax = axes[2]
ax.plot(x, [100 * (1 - intr[c]["vocab_word_initial"]) for c in cs], marker="o", color="C2", label="word-internal tokens in vocab (%)")
ax.set_xticks(x, lab)
ax.set_ylabel("word-internal share of vocabulary (%)", color="C2")
ax3 = ax.twinx()
ax3.plot(x, [intr[c]["test"]["types_used"] for c in cs], marker="^", color="C4")
ax3.set_ylabel("token types used on test", color="C4")
ax.set_title("vocabulary shifts from fragments to whole words")

fig.tight_layout()
fig.savefig(f"{D}/RESULTS.png", dpi=150)
print("wrote RESULTS.png")
