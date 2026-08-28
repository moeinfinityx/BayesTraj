import json
from pathlib import Path

import numpy as np
import pandas as pd

from bayestraj.figures import generate_all
from bayestraj.paper import ROOT, compute_claims


def test_claims_match_submission():
    expected = json.loads((ROOT / "expected" / "claims.json").read_text())
    actual = compute_claims()
    for key, target in expected.items():
        assert np.isclose(actual[key], target, atol=5e-4, rtol=0), key


def test_all_nine_figures_smoke(tmp_path: Path):
    generated = generate_all(tmp_path)
    assert len(generated) == 9
    assert all(path.stat().st_size > 1_000 for path in generated)


def test_readme_dataset_statistics_are_internally_consistent():
    data = ROOT / "data" / "paper_inputs"
    datasets = pd.read_csv(data / "dataset_statistics.csv")
    detail = datasets[datasets.dataset.ne("all")]
    total = datasets[datasets.dataset.eq("all")].iloc[0]
    assert len(detail) == 4
    assert int(detail.evaluated_tasks_per_cell.sum()) == int(total.evaluated_tasks_per_cell) == 2187
    assert int(detail.task_backbone_seed_evaluations.sum()) == int(total.task_backbone_seed_evaluations) == 19683
    assert int(detail.cached_trajectories.sum()) == int(total.cached_trajectories) == 314928
    assert (detail.task_backbone_seed_evaluations == detail.evaluated_tasks_per_cell * 9).all()
    assert (detail.cached_trajectories == detail.task_backbone_seed_evaluations * 16).all()

    success = pd.read_csv(data / "dataset_backbone_success_rates.csv")
    assert len(success) == 12
    assert not success.duplicated(["dataset", "backbone"]).any()
    for seed in (101, 202, 303):
        expected = 100 * success[f"successes_seed{seed}"] / success.evaluated_tasks_per_cell
        assert np.allclose(expected, success[f"success_rate_seed{seed}_pct"], atol=5e-7)
    rates = success[[f"success_rate_seed{seed}_pct" for seed in (101, 202, 303)]]
    assert np.allclose(rates.mean(axis=1), success.success_rate_mean_pct, atol=5e-7)
    assert np.allclose(rates.std(axis=1, ddof=1), success.success_rate_sample_sd_pct, atol=5e-7)

    steps = pd.read_csv(data / "trajectory_step_statistics.csv")
    assert len(steps) == 12
    assert not steps.duplicated(["dataset", "backbone"]).any()
    tasks = detail.set_index("dataset").evaluated_tasks_per_cell
    expected_trajectories = steps.dataset.map(tasks) * 3 * 16
    assert (steps.trajectories == expected_trajectories).all()
    seed_means = steps[[f"seed{seed}_mean_steps" for seed in (101, 202, 303)]]
    assert np.allclose(seed_means.mean(axis=1), steps.mean_steps, atol=5e-10)
    assert (steps.sample_sd_steps >= 0).all()
    assert ((steps.min_steps <= steps.median_steps) & (steps.median_steps <= steps.max_steps)).all()
