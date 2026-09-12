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


def select_top_k(
    group: pd.DataFrame,
    top_k: int,
    sector_map: Dict[str, str] | None,
    broad_min_edge: float = 0.20,
) -> List[str]:
    """按分数贪心选择 Top-K。

    规则:① 一行业一席;② 宽基不是常驻席位——宽基要占席,其最高分必须比
    当期第 K 名行业分数高出 broad_min_edge(默认20%),否则名额全部让给行业。
    """
    if not sector_map:
        return group["ts_code"].head(top_k).tolist()
    is_broad = group["ts_code"].map(lambda c: sector_map.get(c, "") == "宽基")
    industries = group.loc[~is_broad].copy()
    broads = group.loc[is_broad].sort_values("score", ascending=False)

    chosen = industries.head(top_k)
    # 兜底:行业不足 K 个时,先用宽基、再允许行业重复补齐
    if len(chosen) < top_k:
        out = chosen["ts_code"].tolist()
        for _, row in group.iterrows():
            if row["ts_code"] in out:
                continue
            out.append(row["ts_code"])
            if len(out) >= top_k:
                break
        return out
    # 宽基挑战最后一个行业席位:最高分须高出 broad_min_edge
    if not broads.empty:
        fifth_industry_score = float(chosen.iloc[-1]["score"])
        best_broad = broads.iloc[0]
        if float(best_broad["score"]) >= fifth_industry_score * (1.0 + broad_min_edge):
            chosen = chosen.copy()
            chosen.iloc[-1, chosen.columns.get_loc("ts_code")] = best_broad["ts_code"]
    return chosen["ts_code"].tolist()


def compute_weights(
    weighting: str,
    chosen: List[str],
    signal_date,
    etf_vol: pd.DataFrame | None,
    max_weight: float,
) -> np.ndarray:
    """持仓权重:equal=等权;rank=排名衰减;inverse-vol=1/20日波动率(单只上限 max_weight)。"""
    k = len(chosen)
    if weighting == "equal" or etf_vol is None:
        return np.full(k, 1.0 / k)
    if weighting == "rank":
        w = np.array([0.30, 0.25, 0.20, 0.15, 0.10])[:k]
        return w / w.sum()
    # inverse-vol
    vols: List[float] = []
    for code in chosen:
        v = np.nan
        if signal_date in etf_vol.index and code in etf_vol.columns:
            v = float(etf_vol.at[signal_date, code])
        vols.append(v)
    vols_arr = np.array(vols, dtype=float)
    # 年化波动率(raw std 是日频,clip 阈值按年化定义)
    vols_arr = vols_arr * np.sqrt(252.0)
    valid = np.isfinite(vols_arr) & (vols_arr > 0)
    if not valid.any():
        return np.full(k, 1.0 / k)
    inv = np.zeros(k, dtype=float)
    inv[valid] = 1.0 / np.clip(vols_arr[valid], 0.05, None)
    inv[~valid] = float(np.median(inv[valid]))  # 缺波动率的新券按中位数处理
    inv = inv / inv.sum()
    if inv.max() > max_weight:
        inv = np.minimum(inv, max_weight)
        inv = inv / inv.sum()
    return inv


def parse_rebound_ladder(spec: str) -> tuple:
    """解析反弹阶梯参数:"0.12:0.25,0.24:0.50" → ((0.12,0.25),(0.24,0.50));none/空 → ()。"""
    if not spec or spec.lower() in ("none", "off"):
        return tuple()
    tiers = []
    for part in spec.split(","):
        rb, target = part.split(":")
        tiers.append((float(rb), float(target)))
    return tuple(sorted(tiers))


def build_rotation_result(
    pred_df: pd.DataFrame,
    top_k: int,
    freq: str = "signal",
    fee_bps: float = 10.0,
    min_holdings: int = 3,
    sector_map: Dict[str, str] | None = None,
    rebalance_overlap: int = 3,
    rebalance_cooldown: int = 10,
    rebalance_max_gap: int = 63,
    broad_min_edge: float = 0.20,
    weighting: str = "equal",
    max_weight: float = 1.0,
    etf_vol: pd.DataFrame | None = None,
) -> pd.DataFrame:
    fee_rate = fee_bps / 10000.0
    signal_dates = pred_df.index.get_level_values("trade_date").unique().sort_values()
    date_pos = {d: i for i, d in enumerate(signal_dates)}
    rebalance_dates = (
        set(signal_dates) if freq == "daily" else set(get_rebalance_signal_dates(signal_dates, freq))
    )

    rows = []
    current_holdings: List[str] = []
    interval_holdings: List[str] = []
    current_weights = np.zeros(0)
    last_rebalance_pos: int | None = None

    for signal_date, group in pred_df.groupby(level="trade_date"):
        group = group.reset_index().sort_values("score", ascending=False)
        ideal = select_top_k(group, top_k, sector_map, broad_min_edge)

        if freq == "signal":
            # 信号驱动:每天算理想持仓,与当前持仓比重合席位数
            if not interval_holdings:
                do_rebalance = True
            else:
                overlap = len(set(ideal) & set(current_holdings))
                gap = (
                    date_pos[signal_date] - last_rebalance_pos
                    if last_rebalance_pos is not None
                    else rebalance_cooldown
                )
                if gap >= rebalance_max_gap and set(ideal) != set(current_holdings):
                    do_rebalance = True  # 校准保险:超过约一季未调仓且信号有变化
                else:
                    do_rebalance = overlap < rebalance_overlap and gap >= rebalance_cooldown
        else:
            do_rebalance = signal_date in rebalance_dates or not interval_holdings

        if do_rebalance or not interval_holdings:
            new_holdings = ideal
            turnover = (
                len(set(current_holdings) ^ set(new_holdings)) / max(len(new_holdings), 1)
                if current_holdings
                else 1.0
            )
            interval_holdings = new_holdings
            current_weights = compute_weights(weighting, new_holdings, signal_date, etf_vol, max_weight)
            current_holdings = new_holdings.copy()
            rebalanced = 1
            last_rebalance_pos = date_pos[signal_date]
        else:
            turnover = 0.0
            rebalanced = 0

        held = group[group["ts_code"].isin(interval_holdings)]
        w = np.asarray(current_weights, dtype=float)
        if len(held) < min_holdings:
            gross_ret = np.nan
            net_ret = np.nan
        else:
            labels = np.array(
                [
                    held.loc[held["ts_code"] == c, "label"].iloc[0] if (held["ts_code"] == c).any() else np.nan
                    for c in interval_holdings
                ],
                dtype=float,
            )
            ok = np.isfinite(labels)
            if ok.any() and w[ok].sum() > 0:
                gross_ret = float(np.dot(w[ok], labels[ok]) / w[ok].sum())
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
                "weights": ",".join(f"{x:.4f}" for x in w),
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
    parser.add_argument(
        "--rebalance-freq",
        choices=["daily", "monthly", "signal"],
        default="monthly",
        help="monthly=每月首个交易日调仓(默认);signal=信号驱动(重合<阈值才调)",
    )
    parser.add_argument(
        "--rebalance-overlap",
        type=int,
        default=3,
        help="信号模式:重合席位数低于该值才触发调仓(默认3,即至少换2席)",
    )
    parser.add_argument(
        "--rebalance-cooldown",
        type=int,
        default=10,
        help="信号模式:两次调仓最小间隔(交易日,默认10)",
    )
    parser.add_argument(
        "--rebalance-max-gap",
        type=int,
        default=63,
        help="信号模式:超过该交易日数未调仓且信号有变化则强制校准(默认63≈一季)",
    )
    parser.add_argument(
        "--no-sector-dedup",
        dest="sector_dedup",
        action="store_false",
        default=True,
        help="关闭'一行业一席'选券(默认开启)",
    )
    parser.add_argument(
        "--ma-filter",
        choices=["none", "half", "flat"],
        default="flat",
        help="MA 闸门减仓方式(弱市清仓 flat;仅在 --risk-layer ma-gate 下生效)",
    )
    parser.add_argument(
        "--risk-layer",
        choices=["ma-gate", "dd-stop", "none"],
        default="dd-stop",
        help="风控层:dd-stop=组合回撤止损(默认,对策略纸面净值 -10%半仓/-15%清仓);ma-gate=沪深300 MA60 闸门(可选项)",
    )
    parser.add_argument("--dd-half-at", type=float, default=-0.10, help="dd-stop:纸面回撤超过该值降至半仓")
    parser.add_argument("--dd-flat-at", type=float, default=-0.15, help="dd-stop:纸面回撤超过该值清仓")
    parser.add_argument(
        "--rebound-ladder",
        default="0.12:0.25,0.24:0.50,0.36:0.75",
        help="dd-stop 反弹阶梯:清仓区从谷底反弹达档位幅度时建仓至对应仓位(none=关闭)",
    )
    parser.add_argument("--entry-dip", type=float, default=0.02, help="入场限价折扣:信号收盘-2% 挂单等回调")
    parser.add_argument("--entry-window", type=int, default=10, help="限价单有效期(交易日),未成交改市价建仓")
    parser.add_argument("--fundflow-warn", type=float, default=0.10, help="资金流验证器:持仓20日份额流入超过该值标记拥挤警示(仅提示不改决策)")
    parser.add_argument(
        "--broad-min-edge",
        type=float,
        default=0.80,
        help="宽基占席门槛:宽基最高分须比第K名行业分数高80%才可入选(用户定版:不是每个月都买宽基;0=自由竞争)",
    )
    parser.add_argument("--ma-window", type=int, default=60)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument(
        "--weighting",
        choices=["equal", "rank", "inverse-vol"],
        default="inverse-vol",
        help="持仓权重:equal=等权;rank=排名衰减;inverse-vol=波动率倒数(默认,单只上限 --max-weight)",
    )
    parser.add_argument("--max-weight", type=float, default=0.35, help="单只权重上限(inverse-vol 生效)")
    parser.add_argument(
        "--pool",
        choices=["sector", "top2", "all"],
        default="sector",
        help="排序截面:sector=每行业最早上市1只(默认);top2=每行业2只;all=全部候选,扩大截面",
    )
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

    pool_max_per_sector = {"sector": 1, "top2": 2, "all": None}[args.pool]
    universe = get_industry_etf_universe(
        pro,
        end_date=resolved_end_date,
        max_symbols=args.max_symbols,
        strict_mode=args.strict_universe,
        fit_end_date=args.train_end,
        pure_industry_only=args.pure_industry_only,
        max_per_sector=pool_max_per_sector,
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

    sector_map = None
    if args.sector_dedup and "sector_key" in universe.columns:
        sector_map = dict(zip(universe["ts_code"], universe["sector_key"]))

    etf_vol20 = None
    if args.weighting == "inverse-vol":
        close_wide = etf_daily.pivot(index="trade_date", columns="ts_code", values="close")
        etf_vol20 = close_wide.pct_change().rolling(20).std()

    rotation = build_rotation_result(
        pred_df,
        top_k=args.top_k,
        freq=args.rebalance_freq,
        fee_bps=args.fee_bps,
        sector_map=sector_map,
        weighting=args.weighting,
        max_weight=args.max_weight,
        etf_vol=etf_vol20,
        broad_min_edge=args.broad_min_edge,
    )
    rotation.to_csv(output_dir / "strategy_returns.csv")

    # ---- 入场执行表(用户定版:等回调限价建仓) ----
    # 调仓次日对目标组合挂"信号收盘价 × (1 - entry_dip)"的限价单,
    # entry_window 个交易日内未成交则改市价建仓。
    # 依据:68 次调仓实测,等回调限价相对"次日开盘全仓"平均多 +0.62%/次(63% 月份胜出)。
    try:
        close_raw = etf_daily.pivot(index="trade_date", columns="ts_code", values="close")
        rb_last_dates = rotation.index[rotation["rebalanced"] == 1]
        d_sig = rb_last_dates[-1]
        hold_last = [h for h in str(rotation.loc[d_sig, "holdings"]).split(",") if h]
        w_last = [float(x) for x in str(rotation.loc[d_sig, "weights"]).split(",")]
        future_dates = [d for d in close_raw.index if d > d_sig][:args.entry_window]
        sig_closes = [float(close_raw.at[d_sig, c]) if c in close_raw.columns else np.nan for c in hold_last]
        name_map2 = dict(zip(universe["ts_code"], universe["name"]))
        limit_orders = pd.DataFrame(
            {
                "ts_code": hold_last,
                "name": [name_map2.get(c, c) for c in hold_last],
                "weight": w_last,
                "signal_date": [d_sig.date()] * len(hold_last),
                "signal_close": sig_closes,
                "limit_price": [round(c * (1 - args.entry_dip), 3) if np.isfinite(c) else np.nan for c in sig_closes],
                "valid_until": [(future_dates[-1].date() if future_dates else None)] * len(hold_last),
                "fallback": ["10日未成交→市价建仓"] * len(hold_last),
            }
        )
        limit_orders.to_csv(output_dir / "limit_orders.csv", index=False)
    except Exception as exc:  # 执行表只是辅助输出,失败不阻断主流程
        print(f"[limit_orders] 生成失败: {exc}")

    # ---- 资金流验证器监控(文章框架:资金流只做验证器,不改决策) ----
    # 对最新持仓输出 20 日份额变化率;流入>阈值(默认10%)的持仓标"拥挤警示"。
    # 实测:份额20日变化率对未来收益 pooled IC = -0.0655(反向),>10%流入的持仓降权50%
    # 仅在少数事件月有效(20%/30%阈值无效),故只提示、不改选券与权重。
    try:
        pro_ff = get_tushare_pro(token=args.token, proxy_url=args.proxy_url, timeout=30)
        rb_last_dates = rotation.index[rotation["rebalanced"] == 1]
        d_sig = rb_last_dates[-1]
        hold_chk = [h for h in str(rotation.loc[d_sig, "holdings"]).split(",") if h]
        rows_chk = []
        for c in hold_chk:
            try:
                fs = pro.fund_share(ts_code=c, start_date=(d_sig - pd.Timedelta(days=90)).strftime("%Y%m%d"),
                                    end_date=d_sig.strftime("%Y%m%d"))
                if fs is None or fs.empty:
                    continue
                fs["trade_date"] = pd.to_datetime(fs["trade_date"])
                s = fs.sort_values("trade_date").set_index("trade_date")["fd_share"].astype(float)
                s = s[~s.index.duplicated()]
                chg = (s.iloc[-1] / s.iloc[-21] - 1) if len(s) >= 21 else np.nan
                rows_chk.append({
                    "ts_code": c,
                    "name": name_map2.get(c, c),
                    "weight": float(dict(zip(hold_last, w_last)).get(c, np.nan)),
                    "share_chg20": round(float(chg), 4) if np.isfinite(chg) else np.nan,
                    "crowding_warn": ("拥挤警示" if (np.isfinite(chg) and chg >= args.fundflow_warn) else ""),
                })
            except Exception:
                continue
        if rows_chk:
            pd.DataFrame(rows_chk).to_csv(output_dir / "fundflow_check.csv", index=False)
            warn_list = [r for r in rows_chk if r["crowding_warn"]]
            if warn_list:
                print("[fundflow] 拥挤警示:", ", ".join(r["ts_code"] for r in warn_list))
    except Exception as exc:
        print(f"[fundflow_check] 生成失败: {exc}")

    benchmark_ret = build_benchmark_series(benchmark_df)
    benchmark_ret.to_csv(output_dir / "benchmark_returns.csv", header=["benchmark_ret"])

    # ---- 风控层:MA60 市场闸门 / 组合回撤止损(默认 MA 闸门,弱市清仓) ----
    ma_filter_metrics: Dict[str, float] = {}
    raw_net = rotation["net_ret"].fillna(0.0)
    if args.risk_layer == "dd-stop":
        # 组合回撤止损:对策略自身纸面净值计算回撤,破 -dd_half_at 半仓、-dd_flat_at 清仓
        # 反弹阶梯(dd-stop 的另一半):清仓区按"从谷底反弹幅度"分档建仓,替代 0→50→100 跳变
        paper_eq = (1 + raw_net).cumprod()
        paper_dd = paper_eq / paper_eq.cummax() - 1.0
        tiers = parse_rebound_ladder(args.rebound_ladder)
        exposure_list = []
        trough = None
        state = 1.0
        for d, level in zip(paper_dd.values, paper_eq.values):
            if d >= args.dd_half_at:            # 满仓区
                state = 1.0
                trough = None
            elif d < args.dd_flat_at:           # 清仓区:按谷底反弹逐档加回
                trough = level if trough is None else min(trough, level)
                rebound = level / trough - 1.0
                state = 0.0
                for rb, target in tiers:
                    if rebound >= rb:
                        state = max(state, target)
                exposure_list.append(state)
                continue
            else:                               # 半仓区
                state = 0.5
                trough = level if trough is None else min(trough, level)
            exposure_list.append(state)
        exposure = pd.Series(exposure_list, index=rotation.index)
        gate_sig = pd.Series(np.nan, index=rotation.index)  # 无市场闸门信号
    elif args.ma_filter != "none":
        bench_px = benchmark_df.sort_values("trade_date").copy()
        bench_px["ma60"] = bench_px["close"].rolling(args.ma_window).mean()
        bench_px["gate_signal"] = (bench_px["close"] > bench_px["ma60"]).astype(float)
        gate_sig = bench_px.set_index("trade_date")["gate_signal"].reindex(rotation.index).fillna(1.0)
        if args.ma_filter == "flat":
            exposure = (gate_sig > 0).astype(float)
        else:  # half
            exposure = pd.Series(np.where(gate_sig > 0, 1.0, 0.5), index=rotation.index)
    else:
        exposure = pd.Series(1.0, index=rotation.index)
        gate_sig = pd.Series(np.nan, index=rotation.index)

    if args.risk_layer != "none" or args.ma_filter != "none":
        gate_cost = exposure.diff().abs().fillna(0.0) * args.fee_bps / 10000.0
        filtered_ret = raw_net * exposure - gate_cost

        filt_df = pd.DataFrame(
            {
                "trade_date": rotation.index,
                "net_ret_raw": rotation["net_ret"].values,
                "gate_signal": gate_sig.values,
                "exposure": exposure.values,
                "net_ret_filtered": filtered_ret.values,
            }
        )
        filt_df.to_csv(output_dir / "filtered_strategy.csv", index=False)

        f_eq = (1 + filtered_ret).cumprod()
        ma_filter_metrics = {
            "risk_layer": args.risk_layer,
            "ma_filter": args.ma_filter,
            "ma_window": args.ma_window,
            "dd_half_at": args.dd_half_at,
            "dd_flat_at": args.dd_flat_at,
            "filtered_cum_return": float(f_eq.iloc[-1] - 1),
            "filtered_max_drawdown": float((f_eq / f_eq.cummax() - 1).min()),
            "filtered_sharpe": float(
                filtered_ret.mean() / filtered_ret.std() * np.sqrt(252)
            ) if filtered_ret.std() else np.nan,
            "exposure_days_full": int((exposure == 1.0).sum()),
            "exposure_days_derisked": int((exposure < 1.0).sum()),
        }

    metrics = summarize_metrics(ic, rotation["net_ret"].dropna(), benchmark_ret)
    metrics.update(
        {
            "resolved_end_date": resolved_end_date,
            "universe_size": int(len(universe)),
            "rebalance_freq": args.rebalance_freq,
            "weighting": args.weighting,
            "max_weight": args.max_weight,
            "fee_bps": args.fee_bps,
            "pure_industry_only": args.pure_industry_only,
        }
    )
    metrics.update(ma_filter_metrics)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_equity_curve(rotation["net_ret"].dropna(), benchmark_ret, output_dir / "equity_curve.png")

    feature_importance = calc_feature_importance(model, valid_df, features)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    feature_importance.head(20).to_csv(output_dir / "feature_importance_top20.csv", index=False)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
