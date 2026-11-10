#!/usr/bin/env python3
"""
bt portfolio backtest runner (generic).

Fetches daily closes directly from Yahoo's chart API (never yfinance),
builds a bt strategy, runs bt.Backtest and writes results/stats.json.

Pilot defaults (no CLI args):
  tickers        SPY AGG
  weighting      equal weight (bt.algos.WeighEqually) -- SPY 0.5 / AGG 0.5
  rebalancing    monthly (bt.algos.RunMonthly + bt.algos.Rebalance)
  history        ~5 years of daily closes
  initial capital 100000 (passed to bt.Backtest as initial_capital)

Generic use:
  python scripts/backtest.py --tickers SPY AGG QQQ
  python scripts/backtest.py --tickers SPY AGG --weights 0.6 0.4
  python scripts/backtest.py --tickers SPY AGG --years 3 --initial-capital 50000
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bt
import numpy as np
import pandas as pd
import requests

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def fetch_closes(ticker, period1, period2):
    """Fetch daily closes for one ticker from Yahoo's chart API directly."""
    url = YAHOO_CHART_URL.format(ticker=ticker)
    params = {"period1": period1, "period2": period2, "interval": "1d"}
    resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart API error for {ticker}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"Yahoo chart API returned no result for {ticker}")
    result0 = results[0]
    timestamps = result0.get("timestamp") or []
    quotes = (result0.get("indicators") or {}).get("quote") or []
    if not timestamps or not quotes:
        raise RuntimeError(f"Yahoo chart API returned no quotes for {ticker}")
    closes = quotes[0].get("close") or []
    if len(closes) != len(timestamps):
        raise RuntimeError(f"Timestamp/close length mismatch for {ticker}")
    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None).normalize()
    series = pd.Series(closes, index=index, name=ticker, dtype="float64")
    return series


def fetch_prices(tickers, years):
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=years)
    period1 = int(start.timestamp())
    period2 = int(end.timestamp())
    series_list = [fetch_closes(t, period1, period2) for t in tickers]
    data = pd.concat(series_list, axis=1)
    data.columns = list(tickers)
    data = data.sort_index()
    # Drop rows with NaNs across the series (any missing close).
    data = data.dropna(how="any")
    data.index.name = "Date"
    return data


def _serializable(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    # ffn / pandas may hand back objects whose str is the useful form
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return str(value)


def _stats_frame_to_dict(frame):
    """Convert a bt/ffn stats DataFrame/Series to {stat_name: value}."""
    out = {}
    if frame is None:
        return out
    # DataFrame: index = stat names, columns = strategy names -> first column
    if isinstance(frame, pd.DataFrame):
        if frame.shape[1] == 0:
            return out
        col = frame.iloc[:, 0]
        for idx, val in col.items():
            out[str(idx)] = _serializable(val)
        return out
    if isinstance(frame, pd.Series):
        for idx, val in frame.items():
            out[str(idx)] = _serializable(val)
        return out
    if isinstance(frame, dict):
        for k, v in frame.items():
            out[str(k)] = _serializable(v)
        return out
    return out


def extract_result_stats(result, backtest_obj):
    """
    Pull stats from the bt Result object using the library's own figures.
    Returns (raw_stats_dict, sources_used_list).
    Tries, in order:
      1. result.stats
      2. result.prices.calc_stats()  (ffn, the stats engine bt uses)
      3. backtest_obj.stats (if present)
    """
    sources = []
    raw = {}

    # 1. result.stats
    try:
        attr = getattr(result, "stats", None)
        if attr is not None:
            val = attr() if callable(attr) else attr
            d = _stats_frame_to_dict(val)
            if d:
                raw = d
                sources.append("result.stats")
                return raw, sources
    except Exception as exc:  # noqa: BLE001
        print(f"extract: result.stats failed: {exc}", file=sys.stderr)

    # 2. prices.calc_stats()
    try:
        prices = getattr(result, "prices", None)
        if prices is not None and hasattr(prices, "calc_stats"):
            gs = prices.calc_stats()
            frame = getattr(gs, "stats", None)
            d = _stats_frame_to_dict(frame)
            if d:
                raw = d
                sources.append("result.prices.calc_stats().stats")
                return raw, sources
    except Exception as exc:  # noqa: BLE001
        print(f"extract: prices.calc_stats() failed: {exc}", file=sys.stderr)

    # 3. backtest.stats
    try:
        attr = getattr(backtest_obj, "stats", None)
        if attr is not None:
            val = attr() if callable(attr) else attr
            d = _stats_frame_to_dict(val)
            if d:
                raw = d
                sources.append("backtest.stats")
                return raw, sources
    except Exception as exc:  # noqa: BLE001
        print(f"extract: backtest.stats failed: {exc}", file=sys.stderr)

    return raw, sources


def find_stat(raw, *needles):
    """Return (key, value) for the first stat whose lowercase key contains a needle."""
    lowered = {str(k).lower(): (str(k), v) for k, v in raw.items()}
    for needle in needles:
        for lk, (orig, v) in lowered.items():
            if needle in lk:
                return orig, v
    return None, None


def main():
    parser = argparse.ArgumentParser(description="bt portfolio backtest (Yahoo chart API data)")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "AGG"],
                        help="Tickers to include (default: SPY AGG)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Optional weights per ticker (default: equal weight)")
    parser.add_argument("--years", type=int, default=5,
                        help="Years of daily history to fetch (default: 5)")
    parser.add_argument("--initial-capital", type=float, default=100000.0,
                        help="Initial capital passed to bt.Backtest (default: 100000)")
    parser.add_argument("--output", default="results/stats.json",
                        help="Output JSON path (default: results/stats.json)")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]

    if args.weights is not None:
        if len(args.weights) != len(tickers):
            parser.error("--weights must have the same length as --tickers")
        total = float(sum(args.weights))
        if total <= 0:
            parser.error("--weights must sum to a positive value")
        weights = {t: float(w) / total for t, w in zip(tickers, args.weights)}
        weighting_label = "specified (WeighSpecified, normalised)"
        weigh_algo = bt.algos.WeighSpecified(**weights)
    else:
        weights = {t: 1.0 / len(tickers) for t in tickers}
        weighting_label = "equal (WeighEqually)"
        weigh_algo = bt.algos.WeighEqually()

    print(f"Fetching ~{args.years} years of daily closes for {tickers} via Yahoo chart API...")
    data = fetch_prices(tickers, args.years)
    if data.empty:
        raise RuntimeError("No price data after dropping NaN rows")
    print(f"Data: {len(data)} rows, {data.index[0].date()} to {data.index[-1].date()}")
    print(data.head())
    print(data.tail())

    strategy = bt.Strategy(
        "portfolio_monthly",
        [
            bt.AlgoStack(
                bt.algos.SelectAll(),
                weigh_algo,
                bt.algos.RunMonthly(),
                bt.algos.Rebalance(),
            )
        ],
    )

    # bt.Backtest accepts initial_capital; fall back if an older/newer
    # signature rejects it so the run still completes.
    try:
        backtest_obj = bt.Backtest(strategy, data, initial_capital=args.initial_capital)
    except TypeError as exc:
        print(f"Backtest(initial_capital=...) rejected ({exc}); retrying without it", file=sys.stderr)
        backtest_obj = bt.Backtest(strategy, data)

    result = bt.run(backtest_obj)

    try:
        result.display()
    except Exception as exc:  # noqa: BLE001
        print(f"result.display() failed (non-fatal): {exc}", file=sys.stderr)

    raw_stats, stat_sources = extract_result_stats(result, backtest_obj)
    stat_keys_read = sorted(raw_stats.keys())
    print(f"Stat sources used: {stat_sources}")
    print(f"Stat keys read: {stat_keys_read}")
    for k in stat_keys_read:
        print(f"  stat[{k!r}] = {raw_stats[k]!r}")

    # Map the library's own figures (values in ffn/bt stats are fractions).
    total_key, total_val = find_stat(raw_stats, "total_return", "total returns", "totalreturn")
    cagr_key, cagr_val = find_stat(raw_stats, "cagr")
    dd_key, dd_val = find_stat(
        raw_stats, "max_drawdown", "max drawdown", "maxdd", "drawdown"
    )

    def to_pct(v):
        return None if v is None else float(v) * 100.0

    # Portfolio value from result.prices (bt's own price/value series).
    # NOTE (verified locally with bt 1.4.1): result.prices is normalised to
    # start at 100 regardless of Backtest(initial_capital=...), so the
    # 100000-based final value is that series scaled by
    # initial_capital / first_price. Both raw and scaled values are recorded.
    start_value = None
    final_value = None
    raw_start_price = None
    raw_final_price = None
    prices_source = None
    try:
        prices = getattr(result, "prices", None)
        if prices is not None and len(prices) > 0:
            frame = prices if isinstance(prices, pd.DataFrame) else None
            series = None
            if frame is not None and frame.shape[1] > 0:
                series = frame.iloc[:, 0].dropna()
                prices_source = "result.prices (first/last value of first column), scaled to initial_capital"
            elif isinstance(prices, pd.Series):
                series = prices.dropna()
                prices_source = "result.prices (first/last value), scaled to initial_capital"
            if series is not None and len(series) > 0:
                raw_start_price = _serializable(series.iloc[0])
                raw_final_price = _serializable(series.iloc[-1])
                if raw_start_price:
                    start_value = float(args.initial_capital)
                    final_value = float(raw_final_price) / float(raw_start_price) * float(args.initial_capital)
    except Exception as exc:  # noqa: BLE001
        print(f"prices extraction failed: {exc}", file=sys.stderr)

    output = {
        "tool": "bt",
        "tool_description": "bt - Flexible Backtesting for Python (by pmorissette)",
        "strategy": {
            "name": "portfolio_monthly",
            "algos": ["SelectAll", "WeighEqually" if args.weights is None else "WeighSpecified",
                      "RunMonthly", "Rebalance"],
            "weighting": weighting_label,
            "weights": weights,
            "tickers": tickers,
        },
        "data": {
            "source": "Yahoo chart API direct (query1.finance.yahoo.com/v8/finance/chart/<ticker>), daily closes, no yfinance",
            "tickers": tickers,
            "years_requested": args.years,
            "data_range": {
                "start": data.index[0].strftime("%Y-%m-%d"),
                "end": data.index[-1].strftime("%Y-%m-%d"),
            },
            "rows": int(len(data)),
        },
        "initial_capital_requested": args.initial_capital,
        "start_value": start_value,
        "final_value": final_value,
        "final_value_note": "result.prices is normalised to 100 at start in bt 1.4.1; final_value = raw_final_price / raw_start_price * initial_capital (100000 in the pilot)",
        "raw_result_prices_start": raw_start_price,
        "raw_result_prices_final": raw_final_price,
        "final_value_source": prices_source,
        "total_return_pct": to_pct(total_val) if isinstance(total_val, (int, float)) else None,
        "total_return_stat_key": total_key,
        "cagr_pct": to_pct(cagr_val) if isinstance(cagr_val, (int, float)) else None,
        "cagr_stat_key": cagr_key,
        "max_drawdown_pct": to_pct(dd_val) if isinstance(dd_val, (int, float)) else None,
        "max_drawdown_stat_key": dd_key,
        "stat_sources_used": stat_sources,
        "stat_keys_read": stat_keys_read,
        "raw_result_stats": raw_stats,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
