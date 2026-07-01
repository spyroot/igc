"""Offline tests for the adapter-comparison report contract.

Pins that per-arm ResultBundles round-trip to/from disk, that compare() builds a
rows=arm x cols=metric table with baseline deltas from the ACTUAL bundles, and that the
fair-comparison check flags mismatched model/split/steps/seq-len (so an unfair comparison
is never silently reported). Pure stdlib — no torch.

Author:
Mus mbayramo@stanford.edu
"""

from pathlib import Path

from igc.modules.train.report import ResultBundle, RunManifest, capture_environment, compare


def _bundle(arm_method, rank, metrics, *, data="ds-v1", split="held-out-a", steps=200, seq=1024):
    """A ResultBundle for one arm with a shared (fair) manifest by default."""
    return ResultBundle(
        manifest=RunManifest(
            run_id=f"run-{arm_method}-{rank}", profile="m1_7b_lora",
            model="Qwen/Qwen2.5-7B-Instruct", tokenizer="qwen2.5",
            adapter_method=arm_method, adapter_rank=rank,
            data_manifest=data, eval_split=split, max_steps=steps, seq_len=seq,
        ),
        metrics=metrics,
    )


def test_bundle_round_trip(tmp_path: Path) -> None:
    """write() then read() reconstructs an equal bundle."""
    b = _bundle("rslora", 32, {"recall@1": 0.71, "action_top1": 0.63})
    p = tmp_path / "report.json"
    b.write(str(p))
    back = ResultBundle.read(str(p))
    assert back.arm == "rslora-r32"
    assert back.metrics == {"recall@1": 0.71, "action_top1": 0.63}
    assert back.manifest.model == "Qwen/Qwen2.5-7B-Instruct"


def test_compare_builds_table_and_deltas() -> None:
    """compare() aggregates arms into an arm x metric table with baseline deltas."""
    lora = _bundle("lora", 16, {"recall@1": 0.60, "action_top1": 0.55})
    rslora = _bundle("rslora", 32, {"recall@1": 0.71, "action_top1": 0.52})
    rep = compare([lora, rslora], baseline="lora")

    assert rep.arms == ["lora-r16", "rslora-r32"]
    assert rep.metrics == ["action_top1", "recall@1"]
    assert rep.baseline == "lora-r16"
    assert rep.table["rslora-r32"]["recall@1"] == 0.71
    # rsLoRA beats baseline on recall, regresses on action — both visible in deltas
    assert round(rep.deltas["rslora-r32"]["recall@1"], 2) == 0.11
    assert round(rep.deltas["rslora-r32"]["action_top1"], 2) == -0.03
    assert not rep.fairness_issues


def test_missing_metric_is_none_not_zero() -> None:
    """An arm that didn't report a metric shows None (not a misleading 0)."""
    a = _bundle("lora", 16, {"recall@1": 0.6})
    b = _bundle("dora", 16, {"recall@1": 0.65, "probe_macro_f1": 0.8})
    rep = compare([a, b], baseline="lora")
    assert rep.table["lora-r16"]["probe_macro_f1"] is None
    assert rep.deltas["dora-r16"]["probe_macro_f1"] is None  # can't diff against a missing baseline value


def test_fairness_check_flags_unequal_setup() -> None:
    """Different data manifest / max_steps across arms is flagged as unfair."""
    a = _bundle("lora", 16, {"recall@1": 0.6}, data="ds-v1", steps=200)
    b = _bundle("rslora", 32, {"recall@1": 0.9}, data="ds-v2", steps=2000)  # more data + more steps
    rep = compare([a, b])
    joined = " ".join(rep.fairness_issues)
    assert "data_manifest" in joined and "max_steps" in joined
    md = rep.to_markdown()
    assert "NOT an apples-to-apples comparison" in md


def test_capture_environment_is_public_safe():
    """capture_environment records versions/GPU but no hostname/IP/endpoint."""
    env = capture_environment()
    assert "python" in env and "platform" in env
    assert set(env) & {"torch", "transformers", "peft", "numpy"}  # library keys present (value may be None)
    import socket
    assert socket.gethostname() not in " ".join(str(v) for v in env.values())  # no host leak


def test_enriched_manifest_round_trips(tmp_path: Path) -> None:
    """The comprehensive-capture fields survive write/read."""
    b = ResultBundle(
        manifest=RunManifest(
            run_id="r1", profile="m1_7b_rslora_r32", model="Qwen/Qwen2.5-7B-Instruct",
            argv=["--train", "llm", "--adapter_method", "rslora"],
            started_at="2026-07-01T09:00:00", checkpoint_path="experiments/r1/last.pt",
            environment=capture_environment(),
            dataset={"num_examples": 4191, "source_mix": {"real": 3000, "synthetic": 1191}},
            training={"final_loss": 3.1, "best_eval": 0.72, "epochs_done": 3},
            warnings=["frozen new-token embeddings on tied backbone"],
        ),
        metrics={"recall@1": 0.72},
    )
    p = tmp_path / "report.json"
    b.write(str(p))
    back = ResultBundle.read(str(p))
    assert back.manifest.argv == ["--train", "llm", "--adapter_method", "rslora"]
    assert back.manifest.dataset["source_mix"]["synthetic"] == 1191
    assert back.manifest.training["epochs_done"] == 3
    assert back.manifest.warnings and back.manifest.environment.get("python")


# Author: Mus mbayramo@stanford.edu
