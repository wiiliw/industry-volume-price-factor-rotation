from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_holdings(text: str) -> List[str]:
    return [x for x in str(text).split(",") if x]


def normalize_positive_weights(scores: pd.Series) -> pd.Series:
    weights = scores.clip(lower=0.0)
    if float(weights.sum()) <= 0:
        return weights * 0.0
    return weights / weights.sum()


def format_weight_map(weight_map: Dict[str, float]) -> str:
    if not weight_map:
        return ""
    parts = [f"{k}:{v:.4f}" for k, v in sorted(weight_map.items(), key=lambda x: (-x[1], x[0]))]
    return ",".join(parts)


def main():
    parser = argparse.ArgumentParser(description="导出60日最终版策略的买卖台账")
    parser.add_argument("--base-strategy", default="publish_repo/results/final/industry_theme_latest/strategy_returns.csv")
    parser.add_argument("--signals", default="artifacts/combined_etf_trend_model/asset_signals.csv")
    parser.add_argument("--filter-strategy", default="artifacts/combined_etf_trend_model_market_filter/filtered_strategy.csv")
    parser.add_argument("--output-csv", default="artifacts/combined_etf_trend_model_market_filter/trade_ledger_hgb_flat.csv")
    args = parser.parse_args()

    base = pd.read_csv(args.base_strategy)
    base["trade_date"] = pd.to_datetime(base["trade_date"])

    signals = pd.read_csv(args.signals)
    signals["trade_date"] = pd.to_datetime(signals["trade_date"])

    filt = pd.read_csv(args.filter_strategy)
    filt["trade_date"] = pd.to_datetime(filt["trade_date"])
    filt = filt[["trade_date", "flat_exposure", "hgb_weight_ret_flat"]].copy()

    merged = base.merge(filt, on="trade_date", how="left")

    prev_codes: List[str] = []
    rows = []

    for _, row in merged.iterrows():
        dt = row["trade_date"]
        base_codes = parse_holdings(row["holdings"])
        exposure = float(row.get("flat_exposure", 1.0))

        day_signal = (
            signals.loc[signals["trade_date"] == dt]
            .set_index("ts_code")
            .reindex(base_codes)
            .dropna(subset=["HistGB_prob"], how="all")
        )

        if exposure <= 0 or day_signal.empty:
            current_codes: List[str] = []
            weight_map: Dict[str, float] = {}
        else:
            raw_scores = day_signal["HistGB_prob"] - 0.5
            weights = normalize_positive_weights(raw_scores)
            weights = weights.loc[weights > 0]
            current_codes = weights.index.tolist()
            weight_map = {k: float(v) for k, v in weights.items()}

        buy_codes = sorted(set(current_codes) - set(prev_codes))
        sell_codes = sorted(set(prev_codes) - set(current_codes))
        keep_codes = sorted(set(prev_codes) & set(current_codes))

        rows.append(
            {
                "trade_date": dt.strftime("%Y-%m-%d"),
                "market_filter_on": int(exposure > 0),
                "base_top5": ",".join(base_codes),
                "final_holdings": ",".join(current_codes),
                "weights": format_weight_map(weight_map),
                "buy": ",".join(buy_codes),
                "sell": ",".join(sell_codes),
                "keep": ",".join(keep_codes),
                "final_count": len(current_codes),
                "strategy_ret": float(row.get("hgb_weight_ret_flat", 0.0)),
            }
        )
        prev_codes = current_codes

    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)
    print(args.output_csv)


if __name__ == "__main__":
    main()
