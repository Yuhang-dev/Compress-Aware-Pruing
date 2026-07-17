from __future__ import annotations

import pandas as pd
import pytest

from casafety.fig10_paired_margins import (
    assert_identity_matches,
    build_paired_frame,
    build_quadrant_summary,
)


LAYERS = [24, 28, 32]
TAUS = {24: 10.0, 28: 20.0, 32: 30.0}


def make_long_frame() -> pd.DataFrame:
    values = {
        "dense": [25.0, 15.0],
        "pruned": [10.0, 10.0],
        "remar": [22.0, 21.0],
    }
    rows = []
    for condition, means in values.items():
        for order, mean in enumerate(means):
            rows.append(
                {
                    "condition": condition,
                    "prompt_id": 128 + order,
                    "eval_order": order,
                    "prompt_sha256": f"hash-{order}",
                    "s24": mean - 10.0,
                    "s28": mean,
                    "s32": mean + 10.0,
                    "s_mean": mean,
                }
            )
    return pd.DataFrame(rows)


def test_build_paired_frame_and_quadrants() -> None:
    paired = build_paired_frame(make_long_frame(), layers=LAYERS, taus=TAUS)
    assert list(paired["prompt_id"]) == [128, 129]
    assert list(paired["tau"]) == [20.0, 20.0]
    assert list(paired["dense_above_pruned_below"]) == [1, 0]
    assert list(paired["dense_above_remar_below"]) == [0, 0]
    assert list(paired["pruned_below_remar_above"]) == [1, 1]
    assert not any(column in paired.columns for column in ("prompt", "response", "text"))

    quadrants = build_quadrant_summary(paired)
    failure = quadrants.loc[
        (quadrants["pair"] == "dense_vs_pruned")
        & (quadrants["dense_state"] == "above")
        & (quadrants["comparison_state"] == "below"),
        "count",
    ]
    assert failure.item() == 1
    assert len(quadrants) == 8


def test_pairing_rejects_prompt_hash_mismatch() -> None:
    frame = make_long_frame()
    frame.loc[
        (frame["condition"] == "remar") & (frame["prompt_id"] == 129),
        "prompt_sha256",
    ] = "wrong"
    with pytest.raises(ValueError, match="Prompt hashes differ"):
        build_paired_frame(frame, layers=LAYERS, taus=TAUS)


def test_identity_comparison_reports_mismatch() -> None:
    expected = {
        "dataset_id": "walledai/AdvBench",
        "dataset_config": "<default>",
        "dataset_split": "train",
        "dataset_column": "auto",
        "prompt_count": 128,
        "prompt_numeric_id_sha256": "a",
        "prompt_id_sha256": "b",
        "prompt_content_sha256": "c",
        "prompt_split_sha256": "d",
    }
    actual = dict(expected)
    actual["prompt_split_sha256"] = "different"
    with pytest.raises(ValueError, match="prompt_split_sha256"):
        assert_identity_matches(actual, expected)
