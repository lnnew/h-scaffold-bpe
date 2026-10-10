import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scaffold_margin.bpe import build_tokenizers
from scaffold_margin.direction_a import pool_candidates, with_selected
from scaffold_margin.pretok import count_words
from scaffold_margin.synth import make_corpus


@pytest.mark.parametrize("swaps", [0, 1])
def test_pilot_runs_only_matched_validation_jobs(tmp_path, monkeypatch, swaps):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("quick_script", scripts / "a_quick.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    root, out = tmp_path / "input", tmp_path / "pilot"
    (root / "tok").mkdir(parents=True)
    pool = build_tokenizers(count_words(make_corpus(250, seed=8)), 400)["scaffold"]
    pool_path = root / "tok/apool_pool_400.json"
    pool.save(str(pool_path))
    teacher = tmp_path / "model.pt"
    teacher.write_bytes(b"fixed checkpoint")
    teacher.with_name("results.json").write_text(json.dumps({"meta": {"tokenizer": str(pool_path)}}))
    calls = []

    def execute(cmd, **kwargs):
        calls.append((cmd, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        if Path(cmd[1]).name == "a_marginal_utility.py":
            tok_dir = Path(cmd[cmd.index("--tok_dir") + 1])
            selected = list(pool_candidates(pool))[:64]
            for method in ("apool_freq", "apool_lm", "apool_randomswap0"):
                with_selected(pool, selected, method).save(str(tok_dir / (method + "_320.json")))
            (out / "selection.json").write_text(json.dumps({"teachers": {"torch:" + str(teacher): {"n_swapped_slots": swaps}}}))

    monkeypatch.setattr(script.subprocess, "run", execute)
    monkeypatch.setattr(sys, "argv", [str(scripts / "a_quick.py"), "--root", str(root), "--out", str(out),
                                    "--teacher", str(teacher), "--K", "320"])
    script.main()
    status = json.loads((out / "status.json").read_text())
    if swaps == 0:
        assert status["status"] == "NO_INTERVENTION"
        assert len(calls) == 1
        return
    assert status["status"] == "COMPLETE"
    jobs = [(cmd, gpu) for cmd, gpu in calls if "--lm_args" in cmd]
    assert len(jobs) == 9
    assert {(cmd[cmd.index("--methods") + 1], cmd[cmd.index("--seeds") + 1]) for cmd, _ in jobs} == {
        (method, str(seed)) for method in ("apool_freq", "apool_lm", "apool_randomswap0") for seed in range(3)}
    for cmd, gpu in jobs:
        assert gpu in {"0", "1"}
        assert cmd[cmd.index("--eval_sets") + 1] == "val"
        assert cmd[cmd.index("--max_steps") + 1] == "1500"
        assert cmd[cmd.index("--splits") + 1:cmd.index("--eval_sets")] == ["lm_train:train", "val:val"]
    assert sum(Path(cmd[1]).name == "stats.py" for cmd, _ in calls) == 2
