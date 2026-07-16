from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest
import torch

from casafety.xstest_orbench_eval import (
    PromptCase,
    apply_remar,
    build_summaries,
    classify_refusal,
    load_cases,
    shard_path,
)


def test_official_prefix_classifiers() -> None:
    assert classify_refusal("  I'm sorry, but I cannot help.", "xstest")
    assert classify_refusal("As an AI language model, I cannot help.", "orbench")
    assert not classify_refusal("Here is a safe explanation.", "xstest")
    assert not classify_refusal("Here is a safe explanation.", "orbench")


def test_dataset_loader_records_identity_without_text(tmp_path: Path) -> None:
    xstest = tmp_path / "xstest.csv"
    hard = tmp_path / "hard.csv"
    toxic = tmp_path / "toxic.csv"
    pd.DataFrame(
        [
            {"id": 1, "prompt": "safe example", "type": "type-a", "label": "safe"},
            {"id": 2, "prompt": "unsafe example", "type": "type-b", "label": "unsafe"},
        ]
    ).to_csv(xstest, index=False)
    pd.DataFrame([{"prompt": "hard example", "category": "cat-a"}]).to_csv(
        hard, index=False
    )
    pd.DataFrame([{"prompt": "toxic example", "category": "cat-b"}]).to_csv(
        toxic, index=False
    )
    args = argparse.Namespace(
        xstest_file=xstest,
        orbench_hard_file=hard,
        orbench_toxic_file=toxic,
        limit_per_split=0,
        allow_nonstandard_counts=True,
    )
    cases, identity = load_cases(args)
    assert len(cases) == 4
    serialized = str(identity)
    assert "safe example" not in serialized
    assert identity["splits"]["xstest_safe"]["count"] == 1
    assert identity["label_intersections"]["orbench_hard_vs_toxic"] == 0


class FakeDownProj:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.zeros(2, 3))


class FakeLayer:
    def __init__(self) -> None:
        self.mlp = type("FakeMlp", (), {"down_proj": FakeDownProj()})()


class FakeModel:
    def __init__(self) -> None:
        self.model = type("FakeDecoder", (), {"layers": [FakeLayer()]})()


def test_apply_remar_uses_outer_product(tmp_path: Path) -> None:
    artifact = tmp_path / "repair.pt"
    torch.save(
        {
            "model": "model/id",
            "variant": "adv_decode",
            "layers": [0],
            "solves": {0: {"r_hat": torch.tensor([1.0, 2.0]), "g": torch.tensor([3.0, 4.0, 5.0])}},
            "vector_sha256": "vector",
        },
        artifact,
    )
    model = FakeModel()
    info = apply_remar(model, artifact, "model/id", [0], eta=0.5)
    expected = 0.5 * torch.outer(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight, expected)
    assert info["layers"] == [0]


def test_summary_has_only_complete_full_comparison(tmp_path: Path) -> None:
    cases = []
    split_specs = (
        ("xstest", "safe", False, 250),
        ("xstest", "unsafe", True, 200),
        ("orbench", "hard", False, 1319),
        ("orbench", "toxic", True, 655),
    )
    for benchmark, split, expected_refusal, count in split_specs:
        split_cases = [
            PromptCase(benchmark, split, str(i), f"prompt-{benchmark}-{split}-{i}", "cat", expected_refusal)
            for i in range(count)
        ]
        cases.extend(split_cases)
        rows = []
        for case in split_cases:
            refused = expected_refusal
            rows.append(
                {
                    "condition": "dense",
                    "benchmark": benchmark,
                    "split": split,
                    "record_key": case.record_key,
                    "prompt_sha256": case.prompt_sha256,
                    "source_id_sha256": case.source_id_sha256,
                    "expected_refusal": int(expected_refusal),
                    "refused": int(refused),
                    "correct": 1,
                    "completion_empty": 0,
                    "completion_tokens": 8,
                    "completion_chars": 40,
                    "completion_sha256": "0" * 64,
                }
            )
        path = shard_path(tmp_path, "dense", benchmark, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    summary, categories, comparison = build_summaries(tmp_path, cases, ["dense"])
    assert len(summary) == 4
    assert len(categories) == 4
    assert comparison.iloc[0]["xstest_balanced_accuracy_pct"] == pytest.approx(100.0)
    assert comparison.iloc[0]["orbench_balanced_accuracy_pct"] == pytest.approx(100.0)
