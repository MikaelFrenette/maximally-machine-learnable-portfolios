# CLI Cheat Sheet

## Main Demo

Run the full demo pipeline and regenerate plots:

```bash
bash run_demo.sh
```

This uses:

- [`configs/demo_run.yaml`](/mnt/c/users/user/dropbox/mace/configs/demo_run.yaml)
- [`scripts/run_pipeline.py`](/mnt/c/users/user/dropbox/mace/scripts/run_pipeline.py)
- [`scripts/plot_results.py`](/mnt/c/users/user/dropbox/mace/scripts/plot_results.py)

## Core Commands

Run the full pipeline from a run config:

```bash
python3 scripts/run_pipeline.py --config configs/demo_run.yaml
```

Regenerate plots from existing run artifacts:

```bash
python3 scripts/plot_results.py --config configs/demo_run.yaml
```

Run extraction / feature materialization only:

```bash
python3 scripts/extract_features.py --config configs/demo_run.yaml
```

## What The Pipeline Writes

For:

```yaml
run_name: mace20_demo
```

artifacts are written under:

```text
outputs/mace20_demo/
```

Main files:

- `panel.csv`
- `features.csv`
- `diagnostics.csv`
- `predictions.csv`
- `weights_raw.csv`
- `weights_normalized.csv`
- `summary.csv`
- `trading.csv`
- `trading_summary.csv`
- `trading_yearly_summary.csv`
- `cumulative_returns.png`
- `yearly_sharpe_heatmap.png`
- `run.txt`

## Useful One-Liners

Inspect the latest run log:

```bash
tail -n 50 outputs/mace20_demo/run.txt
```

Inspect the selected iteration:

```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/mace20_demo/diagnostics.csv")
print(df.loc[df["is_selected"] == True].to_string(index=False))
PY
```

Inspect the one-row summary:

```bash
python3 - <<'PY'
import pandas as pd
print(pd.read_csv("outputs/mace20_demo/summary.csv").to_string(index=False))
PY
```

## Testing

Run the full test suite:

```bash
python3 -m pytest -q
```

Run lint:

```bash
ruff check .
```

## Notes

- The canonical public run config is [`configs/demo_run.yaml`](/mnt/c/users/user/dropbox/mace/configs/demo_run.yaml).
- `run_pipeline.py` is the main CLI.
- `plot_results.py` is safe to rerun after a completed pipeline run.
