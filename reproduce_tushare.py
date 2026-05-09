from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from scr.core import Factor_Calculator
from scr.data_service import (
    DEFAULT_PROXY_URL,
    DEFAULT_TOKEN,
    get_industry_etf_universe,
    get_many_etf_prices,
    get_trade_calendar,
    get_ts_index_price,
    get_tushare_pro,
    panelize_price_data,
    resolve_latest_end_date,
)


@dataclass(frozen=True)
class FactorSpec:
    feature_name: str
    method_name: str
    window: int | None = None
    window1: int | None = None
    window2: int | None = None


def build_factor_specs() -> List[FactorSpec]:
    windows = [5, 10, 20, 60, 120, 180]
    simple_factors = {
        "amount_std": "AmountVolatility",
        "volume_std": "VolumeVolatility",
        "long_short": "NetPosition",
        "price_vol_rank_cov": "VolumePriceRankCorr",
        "price_vol_cor": "VolumePriceCorr",
        "price_divergence": "FirstOrderDivergence",
        "price_amf": "VolumeAmplitudeCoMovement",
    }

    specs: List[FactorSpec] = []
    for w1, w2, w3 in permutations(windows, 3):
        specs.append(
            FactorSpec(
                feature_name=f"SencondMomentum_{w1}_{w2}_{w3}",
                method_name="second_order_mom",
                window=w3,
                window1=w1,
                window2=w2,
            )
        )

    for w1, w2 in permutations(windows, 2):
        if w1 <= w2:
            continue
        specs.append(
            FactorSpec(
                feature_name=f"MomentumTermSpread_{w1}_{w2}",
                method_name="diff_period_mom",
                window1=w1,
                window2=w2,
            )
        )
        specs.append(
            FactorSpec(
                feature_name=f"PositionChange_{w1}_{w2}",
                method_name="long_short_pct",
                window1=w1,
                window2=w2,
            )
        )

    for method_name, prefix in simple_factors.items():
        for window in windows:
            specs.append(
                FactorSpec(
                    feature_name=f"{prefix}_{window}",
                    method_name=method_name,
                    window=window,
                )
            )
    return specs


def factor_wide_to_long(frame: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    stacked = frame.stack(dropna=False).rename(feature_name)
    stacked.index.names = ["trade_date", "ts_code"]
    return stacked.to_frame()


def build_feature_frame(price_panel: pd.DataFrame) -> pd.DataFrame:
    calc = Factor_Calculator(price_panel)
    feature_frames: List[pd.DataFrame] = []
    for spec in build_factor_specs():
        factor_df = calc.transform(
            spec.method_name,
            window=spec.window,
            window1=spec.window1,
            window2=spec.window2,
        )
        feature_frames.append(factor_wide_to_long(factor_df, spec.feature_name))
    return pd.concat(feature_frames, axis=1).astype(np.float32)


def build_label_frame(price_panel: pd.DataFrame) -> pd.DataFrame:
    open_df = price_panel["open"]
    label_df = open_df.shift(-2).div(open_df.shift(-1)).sub(1.0)
    stacked = label_df.stack(dropna=False).rename("label")
    stacked.index.names = ["trade_date", "ts_code"]
    return stacked.to_frame().astype(np.float32)


def build_benchmark_series(index_df: pd.DataFrame) -> pd.Series:
    benchmark = (
        index_df.sort_values("trade_date")
        .assign(benchmark_ret=lambda x: x["open"].shift(-1).div(x["open"]).sub(1.0))
        .set_index("trade_date")["benchmark_ret"]
        .dropna()
    )
    benchmark.index = pd.to_datetime(benchmark.index)
    benchmark.name = "benchmark_ret"
    return benchmark


def filter_dates(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    idx = frame.index.get_level_values("trade_date")
    mask = (idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date))
    return frame.loc[mask].copy()


def fit_model(train_valid: pd.DataFrame, features: List[str]) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train_valid[features], train_valid["label"])
    return model


def calc_daily_ic(df: pd.DataFrame) -> pd.Series:
    return df.groupby(level="trade_date").apply(
        lambda x: spearmanr(x["score"], x["label"], nan_policy="omit").statistic
    )


def calc_quantile_returns(df: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    def _assign_quantiles(group: pd.DataFrame) -> pd.Series:
        ranked = group["score"].rank(method="first")
        try:
            return pd.qcut(ranked, quantiles, labels=False) + 1
        except ValueError:
            return pd.Series(index=group.index, dtype="float64")

    out = df.copy()
    out["quantile"] = df.groupby(level="trade_date", group_keys=False).apply(
        _assign_quantiles
    )
    out = out.dropna(subset=["quantile"])
    qret = (
        out.reset_index()
        .pivot_table(index="trade_date", columns="quantile", values="label", aggfunc="mean")
        .sort_index()
    )
    qret.columns = [f"Q{int(c)}" for c in qret.columns]
    if {"Q1", f"Q{quantiles}"} <= set(qret.columns):
        qret["long_short"] = qret[f"Q{quantiles}"] - qret["Q1"]
    return qret


def get_rebalance_signal_dates(signal_dates: pd.Index, freq: str) -> pd.Index:
    if freq == "daily":
        return signal_dates
    periods = signal_dates.to_period("M")
    return signal_dates[~periods.duplicated()]


def build_rotation_result(
    pred_df: pd.DataFrame,
    top_k: int,
    freq: str = "monthly",
    fee_bps: float = 10.0,
    min_holdings: int = 3,
) -> pd.DataFrame:
    fee_rate = fee_bps / 10000.0
    signal_dates = pred_df.index.get_level_values("trade_date").unique().sort_values()
    rebalance_dates = set(get_rebalance_signal_dates(signal_dates, freq))

    rows = []
    current_holdings: List[str] = []
    interval_holdings: List[str] = []

    for signal_date, group in pred_df.groupby(level="trade_date"):
        group = group.reset_index().sort_values("score", ascending=False)
        if signal_date in rebalance_dates or not interval_holdings:
            new_holdings = group["ts_code"].head(top_k).tolist()
            turnover = (
                len(set(current_holdings) ^ set(new_holdings)) / max(len(new_holdings), 1)
                if current_holdings
                else 1.0
            )
            interval_holdings = new_holdings
            current_holdings = new_holdings.copy()
            rebalanced = 1
        else:
            turnover = 0.0
            rebalanced = 0

        held = group[group["ts_code"].isin(interval_holdings)]
        if len(held) < min_holdings:
            gross_ret = np.nan
            net_ret = np.nan
        else:
            gross_ret = float(held["label"].mean())
            net_ret = gross_ret - turnover * fee_rate

        rows.append(
            {
                "trade_date": signal_date,
                "gross_ret": gross_ret,
                "net_ret": net_ret,
                "turnover": turnover,
                "rebalanced": rebalanced,
                "holdings": ",".join(interval_holdings),
            }
        )

    return pd.DataFrame(rows).set_index("trade_date")


def summarize_metrics(
    ic: pd.Series,
    strategy_ret: pd.Series,
    benchmark_ret: pd.Series,
) -> Dict[str, float]:
    aligned_bench = benchmark_ret.reindex(strategy_ret.index).fillna(0.0)
    excess = strategy_ret - aligned_bench
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "ic_ir": float(ic.mean() / ic.std()) if ic.std() else np.nan,
        "strategy_cum_return": float((1.0 + strategy_ret).prod() - 1.0),
        "benchmark_cum_return": float((1.0 + aligned_bench).prod() - 1.0),
        "excess_cum_return": float((1.0 + excess).prod() - 1.0),
        "strategy_avg_daily_return": float(strategy_ret.mean()),
        "strategy_vol_daily": float(strategy_ret.std()),
    }


def plot_equity_curve(strategy_ret: pd.Series, benchmark_ret: pd.Series, output_path: Path) -> None:
    bench = benchmark_ret.reindex(strategy_ret.index).fillna(0.0)
    curve = (1.0 + strategy_ret).cumprod()
    bench_curve = (1.0 + bench).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    curve.plot(ax=ax, label="Strategy")
    bench_curve.plot(ax=ax, label="CSI300")
    ax.set_title("Industry Rotation Strategy vs CSI300")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def calc_feature_importance(
    model,
    valid_df: pd.DataFrame,
    features: List[str],
    sample_size: int = 20000,
    n_repeats: int = 3,
) -> pd.DataFrame:
    if valid_df.empty:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    sample = valid_df[features + ["label"]].dropna(subset=["label"])
    if sample.empty:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=42)
    result = permutation_importance(
        model,
        sample[features],
        sample["label"],
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=1,
    )
    imp = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return imp.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复刻行业有效量价因子与行业轮动策略")
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--valid-end", default="2020-12-31")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--pause-seconds", type=float, default=0.65)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--proxy-url", default=DEFAULT_PROXY_URL)
    parser.add_argument("--cache-dir", default="cache/tushare_daily")
    parser.add_argument("--output-dir", default="artifacts/tushare_repro")
    parser.add_argument("--rebalance-freq", choices=["daily", "monthly"], default="monthly")
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--strict-universe", action="store_true", default=True)
    parser.add_argument("--latest-if-missing", action="store_true", default=True)
    parser.add_argument("--pure-industry-only", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pro = get_tushare_pro(token=args.token, proxy_url=args.proxy_url, timeout=30)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    if args.end_date:
        resolved_end_date = args.end_date
    elif args.latest_if_missing:
        resolved_end_date = resolve_latest_end_date(pro, today=today)
    else:
        resolved_end_date = today

    universe = get_industry_etf_universe(
        pro,
        end_date=resolved_end_date,
        max_symbols=args.max_symbols,
        strict_mode=args.strict_universe,
        fit_end_date=args.train_end,
        pure_industry_only=args.pure_industry_only,
    )
    universe.to_csv(output_dir / "universe.csv", index=False)

    etf_daily = get_many_etf_prices(
        pro,
        universe,
        start_date=args.start_date,
        end_date=resolved_end_date,
        cache_dir=cache_dir,
        pause=args.pause_seconds,
    )
    etf_daily.to_csv(output_dir / "etf_daily.csv", index=False)

    benchmark_df = get_ts_index_price(
        pro,
        ts_code=args.benchmark,
        start_date=args.start_date,
        end_date=resolved_end_date,
        pause=args.pause_seconds,
    )
    benchmark_df.to_csv(output_dir / "benchmark_daily.csv", index=False)

    price_panel = panelize_price_data(etf_daily)
    feature_df = build_feature_frame(price_panel)
    label_df = build_label_frame(price_panel)
    dataset = feature_df.join(label_df, how="inner").dropna(subset=["label"])
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset.to_pickle(output_dir / "dataset.pkl")

    train_df = filter_dates(dataset, args.start_date, args.train_end)
    valid_df = filter_dates(
        dataset,
        (pd.Timestamp(args.train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        args.valid_end,
    )
    test_df = filter_dates(
        dataset,
        (pd.Timestamp(args.valid_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        resolved_end_date,
    )

    features = [c for c in dataset.columns if c != "label"]
    train_valid_df = pd.concat([train_df, valid_df]).sort_index()
    model = fit_model(train_valid_df, features)

    pred_df = test_df[["label"]].copy()
    pred_df["score"] = model.predict(test_df[features]).astype(np.float32)
    pred_df.to_pickle(output_dir / "predictions.pkl")
    pred_df.to_csv(output_dir / "predictions.csv")

    ic = calc_daily_ic(pred_df).dropna()
    ic.to_csv(output_dir / "daily_ic.csv", header=["ic"])

    quantile_returns = calc_quantile_returns(pred_df)
    quantile_returns.to_csv(output_dir / "quantile_returns.csv")

    rotation = build_rotation_result(
        pred_df,
        top_k=args.top_k,
        freq=args.rebalance_freq,
        fee_bps=args.fee_bps,
    )
    rotation.to_csv(output_dir / "strategy_returns.csv")

    benchmark_ret = build_benchmark_series(benchmark_df)
    benchmark_ret.to_csv(output_dir / "benchmark_returns.csv", header=["benchmark_ret"])

    metrics = summarize_metrics(ic, rotation["net_ret"].dropna(), benchmark_ret)
    metrics.update(
        {
            "resolved_end_date": resolved_end_date,
            "universe_size": int(len(universe)),
            "rebalance_freq": args.rebalance_freq,
            "fee_bps": args.fee_bps,
            "pure_industry_only": args.pure_industry_only,
        }
    )
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_equity_curve(rotation["net_ret"].dropna(), benchmark_ret, output_dir / "equity_curve.png")

    feature_importance = calc_feature_importance(model, valid_df, features)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    feature_importance.head(20).to_csv(output_dir / "feature_importance_top20.csv", index=False)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
