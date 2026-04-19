from __future__ import annotations

import pandas as pd

from mmlp.analysis import summarize_mace_run
from mmlp.model.mace import FittedMaceModel, MacePanelMatrix


def test_summarize_mace_run_returns_expected_metrics() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03],
            "B": [0.02, 0.01, 0.00],
        },
        index=dates,
    )
    panel_matrix = MacePanelMatrix(
        dates=tuple(dates),
        asset_ids=("A", "B"),
        returns=returns,
    )
    fitted_model = FittedMaceModel(
        intercept_=0.0,
        weights_=pd.Series({"A": 0.6, "B": 0.4}),
        z1_=pd.Series([0.0, 0.1, 0.2], index=dates),
        z2_=pd.Series([0.0, 0.1, 0.2], index=dates),
        diagnostics_=pd.DataFrame(
            {
                "iteration": [1, 2],
                "latent_delta_rmse": [0.5, 0.2],
            }
        ),
        selected_iteration_=2,
        selection_rule_="last_iteration",
        selection_metric_="iteration",
        selection_score_=2.0,
        ridge_model_=object(),
        random_forest_model_=object(),
    )

    summary = summarize_mace_run(fitted_model=fitted_model, panel_matrix=panel_matrix)

    assert len(summary) == 1
    assert {
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "effective_n",
    } <= set(summary.columns)
    assert float(summary.loc[0, "selected_latent_delta_rmse"]) == 0.2
