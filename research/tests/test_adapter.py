"""Cross-check this repo's segmentation against the team's hscaffold_bpe on the team's REAL tokenizers.

Skipped unless the team's repo is importable (set HSCAFFOLD_REPO=/path/to/h-scaffold-bpe)."""
import os, sys
import pytest

from pathlib import Path

# Inside the team repo (branch jihyun/research) the team package is the repo root: research/tests/ -> parents[2]
REPO = os.environ.get("HSCAFFOLD_REPO", str(Path(__file__).resolve().parents[2]))
pytestmark = pytest.mark.skipif(not os.path.isdir(f"{REPO}/artifacts/tokenizers"), reason="team repo not available")

TEXT = """The quick brown fox jumps over the lazy dog. I ate a cookie; the charger was referencing Gerrothorax in Croatia!
Husbands consuming destructible Oscillation... Arizona (Ari+zona), WikiText-103: 1,234.56 = 3 x 10^4 -- "quotes" and 'apostrophes'.
Tokenization of rare words: Rifaximin, Murdoch, Verdun, Gamelan, structuralism, descendants, peroration, kindness.
   leading spaces,	tabs, and newlines
 end. 한국어 문장도 섞어 봅니다. Привет, мир! 日本語のテキスト。 emoji 🙂 and zero-width​joiner.
"""


def _team(variant, K, backend="python"):
    sys.path.insert(0, f"{REPO}/src")
    from hscaffold_bpe.tokenizer import HierarchicalTokenizer
    return HierarchicalTokenizer.from_file(f"{REPO}/artifacts/tokenizers/{variant}-{K}.json", ) if backend == "auto" else \
        HierarchicalTokenizer(__import__("hscaffold_bpe.model", fromlist=["TokenizerModel"]).TokenizerModel.load(
            f"{REPO}/artifacts/tokenizers/{variant}-{K}.json"), backend=backend)


def _bytes_team(tok, ids):
    return [tok.model.tokens[tok._model_to_internal[i]] for i in ids]


@pytest.mark.parametrize("variant", ["bpe", "scaffold", "hierarchical"])
@pytest.mark.parametrize("K", [32768, 65536])
def test_same_segmentation_as_team_implementation(variant, K):
    from scaffold_margin.adapter import load_team_tokenizer
    team = _team(variant, K)
    mine = load_team_tokenizer(f"{REPO}/artifacts/tokenizers/{variant}-{K}.json", cache=False)
    for line in TEXT.split("\n"):
        t_ids = team.encode(line)
        m_ids = mine.encode(line)
        assert _bytes_team(team, t_ids) == [mine.token_bytes[mine.from_model[i]] for i in m_ids], line
    assert mine.n_normal + 1 == team.vocab_size  # + <|endoftext|>


def test_tokenizer_load_autodetects_team_format_and_uses_team_pattern():
    from scaffold_margin.bpe import Tokenizer
    t = Tokenizer.load(f"{REPO}/artifacts/tokenizers/hierarchical-32768.json", cache=False)
    assert t.mode == "hscaffold" and t.pattern == "gpt2"
    team = _team("hierarchical", 32768)
    for s in ["日本語のテキストと한국어 문장, naïve café — ok", "ก็ได้ครับ नमस्ते दुनिया"]:
        assert _bytes_team(team, team.encode(s)) == [t.token_bytes[t.from_model[i]] for i in t.encode(s)], s


def test_find_tokenizer_and_residuals():
    from scaffold_margin.adapter import attach_residuals, find_tokenizer
    from scaffold_margin.bpe import Tokenizer
    from scaffold_margin.pretok import count_words
    p = find_tokenizer(f"{REPO}/artifacts/tokenizers", "hscaffold", 32768)
    assert p.endswith("hierarchical-32768.json")
    t = attach_residuals(Tokenizer.load(p), count_words(TEXT.split("\n"), "gpt2"))
    assert sum(t.meta["resid"]) > 0 and len(t.meta["resid"]) == len(t.levels)


@pytest.mark.parametrize("K,tau", [(800, 0.25), (800, 0.5), (800, 0.75), (1500, 0.5)])
def test_trainer_reproduces_team_trainer_exactly(K, tau, tmp_path):
    """Same counts -> identical merge strings, L0 set and L1/L2 levels as hscaffold_bpe.trainer (hierarchical)."""
    sys.path.insert(0, f"{REPO}/src")
    from hscaffold_bpe.trainer import train_from_counts
    from scaffold_margin.adapter import load_team_tokenizer
    from scaffold_margin.bpe import assign_levels, train_bpe
    from scaffold_margin.pretok import count_words
    from scaffold_margin.synth import make_corpus

    wc = count_words(make_corpus(1500, words_per_doc=200, n_stems=3000, seed=1, stem_seed=0), "gpt2")
    team_model = train_from_counts(wc, K + 1, "hierarchical", show_progress=False, level2_threshold=tau)
    team_model.save(str(tmp_path / "t.json"))
    t = load_team_tokenizer(str(tmp_path / "t.json"))
    team = {t.token_bytes[256 + i]: int(t.levels[256 + i]) for i in range(len(t.merges))}
    r = train_bpe(wc, target_normal=K, log_every=0)                    # scaffold_rule="global" (default)
    lv = assign_levels(r, tau, K)
    tb = [bytes([i]) for i in range(256)]
    for a, b in r.merges:
        tb.append(tb[a] + tb[b])
    mine = {tb[256 + i]: int(lv[256 + i]) for i in range(len(r.merges))}
    assert team == mine


def test_team_trainer_pool_has_requested_size():
    from scaffold_margin.adapter import team_pool_tokenizer
    from scaffold_margin.pretok import count_words
    from scaffold_margin.synth import make_corpus

    wc = count_words(make_corpus(400, seed=1, stem_seed=0), "gpt2")
    assert team_pool_tokenizer(wc, 600, REPO).n_normal == 600
