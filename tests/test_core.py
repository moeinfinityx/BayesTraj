import numpy as np

from bayestraj.core import (
    LinearGaussianModel,
    calibrate_variance_threshold,
    count_entropy_prior,
    entropy_grid,
    fuse_on_grid,
    lcb_score,
    select_stopping_time,
)


def test_grid_and_count_prior_are_valid():
    grid = entropy_grid(16, 257)
    assert len(grid) == 257 and grid[0] == 0 and np.isclose(grid[-1], np.log(17))
    mean, variance = count_entropy_prior([3, 1, 1], draws=4096, seed=7)
    assert 0 < mean < np.log(4)
    assert variance > 0


def test_linear_gaussian_fusion_normalizes():
    rng = np.random.default_rng(8)
    target = rng.uniform(0, 2, 100)
    features = np.column_stack([target, .5 * target]) + rng.normal(0, .1, (100, 2))
    model = LinearGaussianModel.fit(features, target)
    mean, variance, posterior = fuse_on_grid(1.0, .3, [1.2, .6], model)
    assert np.isclose(posterior.sum(), 1)
    assert 0 <= mean <= np.log(17) and variance >= 0
    assert lcb_score(mean, variance) <= mean


def test_first_certificate_and_calibration():
    variances = {2: .5, 3: .2, 4: .08, 5: .03}
    assert select_stopping_time(variances, budget=6, threshold=.1, width=4) == 4
    assert select_stopping_time(variances, budget=6, threshold=.01, width=4) == 6
    threshold = calibrate_variance_threshold([variances] * 5, 6, rho=.8, width=4)
    assert threshold in variances.values()

