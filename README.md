# MACE

Python **attempt** to replicate the paper *Maximally Machine-Learnable Portfolios* by **Philippe Goulet Coulombe** and **Maximilian Göbel** (April 4, 2024 draft). This repository is **not** the official paper repository. The implementation should be read as a clean, public engineering reproduction effort, not as a claim of exact numerical parity with the authors’ results.

## Objective

The model’s objective is to learn a synthetic portfolio return that is as predictable as possible by alternating between:

- a **linear portfolio-weight estimation step**
- a **nonlinear forecasting step** on the portfolio’s own lagged dynamics

More precisely, the paper frames MACE as a **collaborative machine learning algorithm** that combines:

- a **constrained Ridge Regression** on the portfolio side
- a **Random Forest** on the forecasting side

Fundamentally, the portfolio process can be written as

$$
z_t = c + r_t^\top w
$$

where:

- $r_t$ is the cross-section of asset returns at time $t$
- $w$ is the portfolio weight vector
- $z_t$ is the synthetic return series that the model tries to make maximally predictable

The repository then evaluates that learned portfolio with a mean-variance timing overlay of the form

$$
\omega_{t+1} = \frac{1}{\gamma}\frac{\hat y_{t+1}}{\hat \sigma^2_{t+1}}
$$

with bounded positions, following the core idea of the paper and the original R implementation.

This matches the paper’s economic framing: maximizing a portfolio’s predictability is directly connected to mean-variance investor utility when the portfolio is traded as a whole.

## Repository Layout

- [`src/mmlp/`](src/mmlp/): package code
- [`scripts/`](scripts/): CLI entrypoints
- [`configs/`](configs/): validated YAML run configs
- [`tests/`](tests/): automated tests
- [`docs/`](docs/): notes and supporting docs
- [`MMLP/`](MMLP/): legacy R reference implementation
- [`data/`](data/): local datasets, including the paper dataset used by the demo

## Data

This repository includes the paper dataset at:

- [`data/daily__MACE_paper.csv`](data/daily__MACE_paper.csv)

The legacy reference notes describe this file as a universe of roughly 200 large-cap stocks sorted in decreasing order of market capitalization, based on the original paper workflow.

More specifically, the legacy notes describe it as a late-2022 NASDAQ snapshot whose constituents were ordered by market capitalization, with the paper workflow using the largest names from that ranking.

The repo also supports rebuilding experiments from **Yahoo Finance** instead of the attached paper CSV. That path is driven by the run config and uses the same pipeline CLI.

## Quick Start

Clone the repository:

```bash
git clone <your-github-url>
cd mace
```

Install the package in your environment:

```bash
pip install -e .
```

Run the end-to-end demo:

```bash
bash run_demo.sh
```

That command uses:

- [`configs/demo_run.yaml`](configs/demo_run.yaml)
- [`scripts/run_pipeline.py`](scripts/run_pipeline.py)
- [`scripts/plot_results.py`](scripts/plot_results.py)

## Main CLI Commands

Run the full pipeline:

```bash
python3 scripts/run_pipeline.py --config configs/demo_run.yaml
```

Regenerate plots from existing artifacts:

```bash
python3 scripts/plot_results.py --config configs/demo_run.yaml
```

Run extraction / feature materialization only:

```bash
python3 scripts/extract_features.py --config configs/demo_run.yaml
```

## Using Yahoo Instead Of The Attached Dataset

The demo config currently points to the attached paper CSV. To run from Yahoo instead, switch the dataset section in your run config to something like:

```yaml
dataset:
  provider: yahoo
  calendar: XNYS
  price_field: "Adj Close"
  start_date: 2003-01-01
  end_date: 2026-03-26
  drop_missing: false
  auto_adjust: false
  progress: false
  universe:
    path: mace_universe.json
    size: 20
```

Then run:

```bash
python3 scripts/run_pipeline.py --config <your_config>.yaml
```

This is the cleaner option if you want a fully rebuildable public-data workflow instead of relying on the attached paper CSV.

## Outputs

For a run with:

```yaml
run_name: mace20_demo
```

artifacts are written to:

```text
outputs/mace20_demo/
```

Typical files:

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

## Example Diagnostics To Inspect

After a run, useful files are:

- [`outputs/mace20_demo/diagnostics.csv`](outputs/mace20_demo/diagnostics.csv)
- [`outputs/mace20_demo/summary.csv`](outputs/mace20_demo/summary.csv)
- [`outputs/mace20_demo/run.txt`](outputs/mace20_demo/run.txt)

The diagnostics log:

- selected ridge alpha
- selected in-sample ridge $R^2$
- target $R^2$
- gap to target
- OOB metric
- latent update size across iterations

## Development

Core checks:

```bash
ruff check .
python3 -m pytest -q
```

There is also a compact command reference at:

- [`CLI_CHEATSHEET.md`](CLI_CHEATSHEET.md)

## Acknowledgment

This project is based on the ideas and legacy R workflow associated with:

- **Goulet Coulombe, Philippe**
- **Göbel, Maximilian**

The original R reference materials retained in [`MMLP/`](MMLP/) were helpful for understanding the intended experiment design, dataset conventions, trading overlay, and post-fit evaluation flow.
