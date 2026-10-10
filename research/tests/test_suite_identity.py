import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/run_lm_suite.py"
    spec = importlib.util.spec_from_file_location("suite_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    tok = tmp_path / "freq_320.json"
    tok.write_text("first tokenizer")
    for split in ("lm_train", "val", "test"):
        (tmp_path / (split + ".jsonl")).write_text('{"text":"original text"}\n')
    calls = []

    def execute(cmd, **kwargs):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / ("meta.json" if "--tok" in cmd else "results.json")).write_text("{}")

    monkeypatch.setattr(script.subprocess, "run", execute)
    args = [str(path), "--tok_dir", str(tmp_path), "--text_dir", str(tmp_path), "--K", "320", "--setting", "pilot",
            "--methods", "freq", "--seeds", "0", "--gpus", "1", "--data_root", str(tmp_path / "data"),
            "--runs", str(tmp_path / "runs"), "--lm_args", "--max_steps", "10"]

    def run(extra=None):
        monkeypatch.setattr(sys, "argv", args if extra is None else args[:-1] + [extra])
        script.main()
    return run, tok, calls


def test_same_experiment_skips_completed_run(experiment):
    run, _, calls = experiment
    run(); run()
    assert len(calls) == 2


def test_budget_change_cannot_reuse_result(experiment):
    run, _, _ = experiment
    run()
    with pytest.raises(ValueError, match="identity changed"):
        run("20")


def test_tokenizer_change_cannot_reuse_stream(experiment):
    run, tok, _ = experiment
    run(); tok.write_text("different tokenizer")
    with pytest.raises(ValueError, match="identity changed"):
        run()
