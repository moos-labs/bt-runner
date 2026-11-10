# bt-runner

Runs a [bt](https://github.com/pmorissette/bt) portfolio backtest in GitHub Actions — commands are supplied through the `COMMANDS` repository variable, the workflow is dispatched manually, and results are downloaded from the `results` artifact.

**bt** is a flexible backtesting framework for Python by pmorissette. It is built around reusable *algos* (algorithm blocks) that are chained into strategies — for example select securities, assign weights, run on a schedule, and rebalance — and then run against historical price data via `bt.Backtest` / `bt.run`.

## Pattern

This repo follows the same runner pattern as `vectorbt-runner` and `finbert-runner`:

1. Edit the **`COMMANDS`** repository variable (Settings → Secrets and variables → Actions → Variables, or via the GitHub API):

   ```
   mkdir -p results
   python scripts/backtest.py 2>&1 | tee results/run.log
   ```

2. **Dispatch the workflow manually** — Actions tab → *Run analysis* → *Run workflow*, or via the GitHub API (`workflow_dispatch` on `run.yml`). The workflow has no other triggers.
3. **Download the `results` artifact** from the completed run. It contains `results/stats.json` and `results/run.log`.

The workflow (`.github/workflows/run.yml`) checks out the repo, sets up Python 3.11, runs `pip install -r requirements.txt`, executes `${{ vars.COMMANDS }}`, and always uploads `results/` as the `results` artifact.

## Pilot backtest

`scripts/backtest.py` with no arguments runs the pilot:

- **Tickers:** `SPY` and `AGG`
- **Data:** ~5 years of daily closes fetched **directly from Yahoo's chart API** (`https://query1.finance.yahoo.com/v8/finance/chart/<ticker>?period1=<epoch>&period2=<epoch>&interval=1d`, browser User-Agent) with `requests` — never yfinance. Closes are parsed into a pandas DataFrame (columns = tickers, DatetimeIndex); rows with NaNs across the series are dropped.
- **Strategy:** `AlgoStack(SelectAll, WeighEqually, RunMonthly, Rebalance)` — **equal weight** (SPY 0.5 / AGG 0.5), rebalanced monthly.
- **Capital:** 100000 passed to `bt.Backtest` as `initial_capital`.
- **Output:** `results/stats.json` with the data range, row count, total return %, CAGR and max drawdown taken from the bt `Result` object's own stats (the exact stat keys read are recorded in the JSON, together with all raw stats), plus the final portfolio value from `result.prices`.

## Generic use

Tickers and weights are CLI arguments — nothing is hardcoded beyond the pilot defaults:

```bash
python scripts/backtest.py --tickers SPY AGG QQQ
python scripts/backtest.py --tickers SPY AGG --weights 0.6 0.4
python scripts/backtest.py --tickers SPY AGG --years 3 --initial-capital 50000
python scripts/backtest.py --help
```

With `--weights`, the strategy uses `WeighSpecified` (weights are normalised to sum to 1); without it, `WeighEqually`.

## Requirements

```
bt
pandas
numpy
requests
```
