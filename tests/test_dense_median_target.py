import math

import torch

from casafety.closed_form_readout_repair import (
    SolveConfig,
    dense_median_target_margins,
    parse_solve_configs,
)


def test_dense_median_target_is_computed_per_layer() -> None:
    dense = {
        24: {"s": torch.tensor([10.0, 20.0, 30.0, 40.0])},
        28: {"s": torch.tensor([20.0, 30.0, 40.0, 50.0])},
    }

    medians = dense_median_target_margins(dense, layers=[24, 28])

    assert medians == {24: 25.0, 28: 35.0}
    margins = {24: medians[24] - 20.0, 28: medians[28] - 30.0}
    config = SolveConfig("tmdense_lb1", 5.0, 1.0, "dense_median", margins)
    assert config.target_margin_by_layer == {24: 5.0, 28: 5.0}


def test_dense_mode_removes_target_margin_sweep() -> None:
    configs = parse_solve_configs("2,6,12,20", "1,5", "dense_median")

    assert [config.solve_id for config in configs] == ["tmdense_lb1", "tmdense_lb5"]
    assert all(math.isnan(config.target_margin) for config in configs)


def test_fixed_mode_preserves_historical_grid() -> None:
    configs = parse_solve_configs("2,6", "1", "fixed")

    assert [config.solve_id for config in configs] == ["tm2_lb1", "tm6_lb1"]
    assert [config.target_margin for config in configs] == [2.0, 6.0]
