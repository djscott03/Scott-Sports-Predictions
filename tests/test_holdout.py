import numpy as np
import pandas as pd

from holdout import holdout, edge_buckets


def _bt(seed=0, n=4000, model_noise=6.0):
    # outcome noise (sd 13) swamps everything at small n; 4000 games keeps the weight estimate stable
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 6, n)
    return pd.DataFrame(dict(season=np.repeat([2021, 2022, 2023, 2024], n // 4),
                             margin=truth + rng.normal(0, 13, n),
                             market_margin=truth + rng.normal(0, 1.0, n),
                             model_margin=truth + rng.normal(0, model_noise, n)))


def test_perfect_market_prefers_weight_one():
    r = holdout(_bt(), [2021, 2022], [2023, 2024], current_w=0.7)
    assert r["best_w"] >= 0.85          # optimum is 36/37 ≈ 0.97; grid + noise allow a step
    assert r["test_mae_market"] <= r["test_mae_current"]


def test_informative_model_earns_weight():
    r = holdout(_bt(model_noise=0.5), [2021, 2022], [2023, 2024], current_w=0.7)
    assert r["best_w"] <= 0.7


def test_edge_buckets_partition_games():
    b = _bt()
    eb = edge_buckets(b, 0.7)
    assert eb.n.sum() <= len(b) and (eb.ats.between(0, 1)).all()
