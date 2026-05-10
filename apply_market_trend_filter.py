from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scr.data_service import (
    get_ts_index_price,
    get_tushare_pro,
    resolve_latest_end_date,
)


def summarize(returns: pd.Series) -> Dict[str, float]:
    ret = returns.fillna(0.0)
    equity = (1.0 + ret).cumprod()
    mdd = float((equity / equity.cummax() - 1.0).min())
    ann = float(equity.iloc[-1] ** (252 / len(ret)) - 1.0)
    ann_vol = float(ret.std() * np.sqrt(252))
    sharpe = float((ret.mean() / ret.std()) * np.sqrt(252)) if ret.std() else 0.0
    return {
        "cum_return": float(equity.iloc[-1] - 1.0),
        "ann_return": ann,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": float((ret > 0).mean()),
    }


def build_filter_frame(start_date: str, end_date: str, ma_window: int) -> pd.DataFrame:
    pro = get_tushare_pro()
    resolved_end = resolve_latest_end_date(pro, end_date)
    idx = get_ts_index_price(pro, "000300.SH", start_date=start_date, end_date=resolved_end)
    idx = idx.sort_values("trade_date").copy()
    idx["ma"] = idx["close"].rolling(ma_window).mean()
    idx["signal"] = (idx["close"] > idx["ma"]).astype(float)
    idx["half_exposure"] = np.where(idx["signal"] > 0, 1.0, 0.5)
    idx["flat_exposure"] = np.where(idx["signal"] > 0, 1.0, 0.0)
    return idx[["trade_date", "close", "ma", "signal", "half_exposure", "flat_exposure"]]


def overlay_returns(
    df: pd.DataFrame,
    return_col: str,
    exposure_col: str,
    fee_rate: float,
) -> pd.Series:
    exposure = df[exposure_col].fillna(1.0)
    turnover = exposure.diff().abs().fillna(exposure.iloc[0])
    return df[return_col].fillna(0.0) * exposure - turnover * fee_rate


def plot_curves(df: pd.DataFrame, cols: Iterable[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in cols:
        (1.0 + df.set_index("trade_date")[col].fillna(0.0)).cumprod().plot(ax=ax, label=col)
    ax.set_title("ETF Trend Model with CSI300 MA Filter")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="对现有策略叠加CSI300均线过滤")
    parser.add_argument("--input-csv", default="artifacts/combined_etf_trend_model/combined_strategy.csv")
    parser.add_argument("--ma-window", type=int, default=60)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--output-dir", default="artifacts/combined_etf_trend_model_market_filter")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combo = pd.read_csv(args.input_csv)
    combo["trade_date"] = pd.to_datetime(combo["trade_date"])

    filt = build_filter_frame(
        start_date=min(args.start_date, combo["trade_date"].min().strftime("%Y-%m-%d")),
        end_date=max(args.end_date, combo["trade_date"].max().strftime("%Y-%m-%d")),
        ma_window=args.ma_window,
    )
    filt["trade_date"] = pd.to_datetime(filt["trade_date"])

    df = combo.merge(filt, on="trade_date", how="left")
    df[["half_exposure", "flat_exposure"]] = df[["half_exposure", "flat_exposure"]].fillna(1.0)
    fee_rate = args.fee_bps / 10000.0

    strategy_cols = [c for c in combo.columns if c.endswith("_ret")]
    metrics: Dict[str, Dict[str, float]] = {}
    plotted = []
    for col in strategy_cols:
        metrics[col] = summarize(df[col])
        plotted.append(col)
        for suffix, exposure_col in [("half", "half_exposure"), ("flat", "flat_exposure")]:
            out_col = f"{col}_{suffix}"
            df[out_col] = overlay_returns(df, col, exposure_col, fee_rate)
            metrics[out_col] = summarize(df[out_col])
            plotted.append(out_col)

    df.to_csv(output_dir / "filtered_strategy.csv", index=False)
    filt.to_csv(output_dir / "market_filter.csv", index=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ma_window": args.ma_window,
                "fee_bps": args.fee_bps,
                "metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_cols = [c for c in ["base_ret", "hgb_weight_ret", "hgb_weight_ret_half", "hgb_weight_ret_flat"] if c in df.columns]
    plot_curves(df, plot_cols, output_dir / "equity_curve.png")
    print(json.dumps({"ma_window": args.ma_window, "fee_bps": args.fee_bps, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
