from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


FEATURES = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_5",
    "vol_10",
    "vol_20",
    "amount_chg_5",
    "volume_chg_5",
    "hl_spread_1",
    "oc_gap_1",
    "dist_ma5",
    "dist_ma10",
    "dist_ma20",
    "close_rank_20",
]


def build_features(etf_daily: pd.DataFrame) -> pd.DataFrame:
    df = etf_daily.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    by_code = df.groupby("ts_code", group_keys=False)
    prev_close = by_code["close"].shift(1)

    df["ret_1"] = by_code["close"].pct_change(1)
    df["ret_3"] = by_code["close"].pct_change(3)
    df["ret_5"] = by_code["close"].pct_change(5)
    df["ret_10"] = by_code["close"].pct_change(10)
    df["ret_20"] = by_code["close"].pct_change(20)

    df["vol_5"] = by_code["close"].pct_change().rolling(5).std().reset_index(level=0, drop=True)
    df["vol_10"] = by_code["close"].pct_change().rolling(10).std().reset_index(level=0, drop=True)
    df["vol_20"] = by_code["close"].pct_change().rolling(20).std().reset_index(level=0, drop=True)

    df["amount_chg_5"] = by_code["amount"].transform(lambda s: s / s.rolling(5).mean() - 1.0)
    df["volume_chg_5"] = by_code["volume"].transform(lambda s: s / s.rolling(5).mean() - 1.0)
    df["hl_spread_1"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["oc_gap_1"] = (df["open"] - prev_close) / prev_close.replace(0, np.nan)

    ma5 = by_code["close"].transform(lambda s: s.rolling(5).mean())
    ma10 = by_code["close"].transform(lambda s: s.rolling(10).mean())
    ma20 = by_code["close"].transform(lambda s: s.rolling(20).mean())
    df["dist_ma5"] = df["close"] / ma5 - 1.0
    df["dist_ma10"] = df["close"] / ma10 - 1.0
    df["dist_ma20"] = df["close"] / ma20 - 1.0

    rolling_min_20 = by_code["close"].transform(lambda s: s.rolling(20).min())
    rolling_max_20 = by_code["close"].transform(lambda s: s.rolling(20).max())
    spread_20 = (rolling_max_20 - rolling_min_20).replace(0, np.nan)
    df["close_rank_20"] = (df["close"] - rolling_min_20) / spread_20

    df["forward_ret"] = by_code["open"].shift(-2) / by_code["open"].shift(-1) - 1.0
    df["label"] = (df["forward_ret"] > 0).astype(int)
    df = df.dropna(subset=FEATURES + ["forward_ret"]).copy()
    return df


def fit_models(train_df: pd.DataFrame) -> Dict[str, object]:
    x_train = train_df[FEATURES]
    y_train = train_df["label"]
    return {
        "HistGB": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=250,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=42,
        ).fit(x_train, y_train),
        "LogReg": LogisticRegression(max_iter=1000).fit(x_train, y_train),
    }


def score_assets(feature_df: pd.DataFrame, train_end: str) -> pd.DataFrame:
    train_df = feature_df.loc[feature_df["trade_date"] <= pd.Timestamp(train_end)].copy()
    test_df = feature_df.loc[feature_df["trade_date"] > pd.Timestamp(train_end)].copy()
    models = fit_models(train_df)
    out = test_df[["trade_date", "ts_code", "forward_ret", "vol_20"]].copy()
    for name, model in models.items():
        out[f"{name}_prob"] = model.predict_proba(test_df[FEATURES])[:, 1]
    return out


def one_way_turnover(prev_holdings: List[str], new_holdings: List[str]) -> float:
    if not prev_holdings:
        return 1.0 if new_holdings else 0.0
    if not new_holdings:
        return 1.0
    return len(set(prev_holdings) ^ set(new_holdings)) / max(len(prev_holdings), len(new_holdings))


def weight_turnover(prev_weights: Dict[str, float], new_weights: Dict[str, float]) -> float:
    keys = set(prev_weights) | set(new_weights)
    if not keys:
        return 0.0
    return 0.5 * sum(abs(prev_weights.get(k, 0.0) - new_weights.get(k, 0.0)) for k in keys)


def get_rebalance_mask(trade_dates: pd.Series, base_rebalanced: pd.Series, freq: str) -> pd.Series:
    if freq == "daily":
        return pd.Series(True, index=trade_dates.index)
    weekly_period = pd.to_datetime(trade_dates).dt.to_period("W-MON")
    weekly_open = ~weekly_period.duplicated()
    return weekly_open | base_rebalanced.fillna(0).astype(int).eq(1)


def get_weighted_return(sub: pd.DataFrame, prob_col: str) -> tuple[float, List[str]]:
    if sub.empty:
        return 0.0, []
    weights = np.clip(sub[prob_col] - 0.5, 0, None)
    if float(weights.sum()) == 0:
        return 0.0, []
    weights = weights / weights.sum()
    return float((sub["forward_ret"] * weights).sum()), sub.index.tolist()


def get_positive_weight_map(sub: pd.DataFrame, prob_col: str) -> Dict[str, float]:
    if sub.empty:
        return {}
    weights = normalize_positive_weights(np.clip(sub[prob_col] - 0.5, 0, None))
    weights = weights.loc[weights > 0]
    return {k: float(v) for k, v in weights.items()}


def normalize_positive_weights(values) -> pd.Series:
    weights = pd.Series(values, index=getattr(values, "index", None), dtype="float64").clip(lower=0.0)
    total = float(weights.sum())
    if total <= 0:
        return weights * 0.0
    return weights / total


def cap_and_renorm(weights: pd.Series, max_weight: float, gross_exposure: float) -> pd.Series:
    if weights.empty or gross_exposure <= 0:
        return weights * 0.0
    capped = weights.clip(upper=max_weight)
    for _ in range(10):
        total = float(capped.sum())
        if total <= 0:
            return capped * 0.0
        if total <= gross_exposure + 1e-12:
            break
        capped = capped * (gross_exposure / total)
        capped = capped.clip(upper=max_weight)
    total = float(capped.sum())
    if total < gross_exposure - 1e-12:
        room = (max_weight - capped).clip(lower=0.0)
        room_sum = float(room.sum())
        if room_sum > 0:
            capped = capped + room / room_sum * min(gross_exposure - total, room_sum)
    return capped


def get_constrained_weighted_return(
    sub: pd.DataFrame,
    prob_col: str,
    prob_threshold: float,
    max_weight: float,
    gross_exposure: float,
    vol_col: str = "vol_20",
) -> tuple[float, List[str], float]:
    if sub.empty:
        return 0.0, [], 0.0
    chosen = sub.loc[sub[prob_col] > prob_threshold].copy()
    if chosen.empty:
        return 0.0, [], 0.0
    base_score = np.clip(chosen[prob_col] - prob_threshold, 0, None)
    vol = chosen[vol_col].replace(0, np.nan)
    risk_adj = base_score / vol
    risk_adj = risk_adj.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(risk_adj.sum()) <= 0:
        return 0.0, [], 0.0
    weights = risk_adj / risk_adj.sum()
    weights = cap_and_renorm(weights, max_weight=max_weight, gross_exposure=gross_exposure)
    if float(weights.sum()) <= 0:
        return 0.0, [], 0.0
    weighted_ret = float((chosen["forward_ret"] * weights).sum())
    return weighted_ret, chosen.index.tolist(), float(weights.sum())


def get_rank_weight_map(sub: pd.DataFrame, prob_col: str, top_n: int) -> Dict[str, float]:
    picked = sub.sort_values(prob_col, ascending=False).head(top_n)
    if picked.empty:
        return {}
    equal_weight = 1.0 / len(picked)
    return {k: equal_weight for k in picked.index.tolist()}


def realized_return(weight_map: Dict[str, float], daily: pd.DataFrame) -> float:
    if not weight_map:
        return 0.0
    ret = 0.0
    for code, weight in weight_map.items():
        if code in daily.index:
            ret += weight * float(daily.at[code, "forward_ret"])
    return ret


def build_combo(
    base_strategy: pd.DataFrame,
    signals: pd.DataFrame,
    fee_bps: float,
    top_n: int,
    prob_threshold: float,
    max_weight: float,
    gross_exposure: float,
    rebalance_freq: str,
) -> pd.DataFrame:
    fee_rate = fee_bps / 10000.0
    signals = signals.copy()
    signals["trade_date"] = pd.to_datetime(signals["trade_date"])
    base_strategy = base_strategy.copy()
    base_strategy["need_rebalance"] = get_rebalance_mask(
        base_strategy["trade_date"],
        base_strategy.get("rebalanced", pd.Series(0, index=base_strategy.index)),
        rebalance_freq,
    )

    prev_hgb_rank_w: Dict[str, float] = {}
    prev_lr_rank_w: Dict[str, float] = {}
    prev_hgb_weight_w: Dict[str, float] = {}
    prev_lr_weight_w: Dict[str, float] = {}
    prev_hgb_cons_w: Dict[str, float] = {}
    prev_lr_cons_w: Dict[str, float] = {}
    rows = []

    for _, row in base_strategy.iterrows():
        dt = pd.Timestamp(row["trade_date"])
        holdings = [x for x in str(row["holdings"]).split(",") if x]
        daily = signals.loc[signals["trade_date"] == dt].set_index("ts_code")
        sub = daily.loc[daily.index.intersection(holdings)].copy()
        do_rebalance = bool(row["need_rebalance"])

        if do_rebalance:
            hgb_rank_w = get_rank_weight_map(sub, "HistGB_prob", top_n)
            lr_rank_w = get_rank_weight_map(sub, "LogReg_prob", top_n)
            hgb_weight_w = get_positive_weight_map(sub, "HistGB_prob")
            lr_weight_w = get_positive_weight_map(sub, "LogReg_prob")

            _, hgb_cons_hold, hgb_cons_expo = get_constrained_weighted_return(
                sub,
                "HistGB_prob",
                prob_threshold=prob_threshold,
                max_weight=max_weight,
                gross_exposure=gross_exposure,
            )
            _, lr_cons_hold, lr_cons_expo = get_constrained_weighted_return(
                sub,
                "LogReg_prob",
                prob_threshold=prob_threshold,
                max_weight=max_weight,
                gross_exposure=gross_exposure,
            )
            hgb_cons_sub = sub.loc[sub.index.intersection(hgb_cons_hold)].copy()
            lr_cons_sub = sub.loc[sub.index.intersection(lr_cons_hold)].copy()
            hgb_cons_base = np.clip(hgb_cons_sub["HistGB_prob"] - prob_threshold, 0, None) / hgb_cons_sub["vol_20"].replace(0, np.nan)
            lr_cons_base = np.clip(lr_cons_sub["LogReg_prob"] - prob_threshold, 0, None) / lr_cons_sub["vol_20"].replace(0, np.nan)
            hgb_cons_weights = normalize_positive_weights(pd.Series(hgb_cons_base, index=hgb_cons_sub.index).replace([np.inf, -np.inf], np.nan).fillna(0.0))
            lr_cons_weights = normalize_positive_weights(pd.Series(lr_cons_base, index=lr_cons_sub.index).replace([np.inf, -np.inf], np.nan).fillna(0.0))
            hgb_cons_weights = cap_and_renorm(hgb_cons_weights, max_weight=max_weight, gross_exposure=gross_exposure)
            lr_cons_weights = cap_and_renorm(lr_cons_weights, max_weight=max_weight, gross_exposure=gross_exposure)
            hgb_cons_w = {k: float(v) for k, v in hgb_cons_weights.loc[hgb_cons_weights > 0].items()}
            lr_cons_w = {k: float(v) for k, v in lr_cons_weights.loc[lr_cons_weights > 0].items()}
        else:
            hgb_rank_w = prev_hgb_rank_w.copy()
            lr_rank_w = prev_lr_rank_w.copy()
            hgb_weight_w = prev_hgb_weight_w.copy()
            lr_weight_w = prev_lr_weight_w.copy()
            hgb_cons_w = prev_hgb_cons_w.copy()
            lr_cons_w = prev_lr_cons_w.copy()
            hgb_cons_expo = float(sum(hgb_cons_w.values()))
            lr_cons_expo = float(sum(lr_cons_w.values()))

        hgb_rank_ret = realized_return(hgb_rank_w, daily)
        lr_rank_ret = realized_return(lr_rank_w, daily)
        hgb_weight_ret = realized_return(hgb_weight_w, daily)
        lr_weight_ret = realized_return(lr_weight_w, daily)
        hgb_cons_ret = realized_return(hgb_cons_w, daily)
        lr_cons_ret = realized_return(lr_cons_w, daily)

        hgb_rank_turn = weight_turnover(prev_hgb_rank_w, hgb_rank_w) if do_rebalance else 0.0
        lr_rank_turn = weight_turnover(prev_lr_rank_w, lr_rank_w) if do_rebalance else 0.0
        hgb_weight_turn = weight_turnover(prev_hgb_weight_w, hgb_weight_w) if do_rebalance else 0.0
        lr_weight_turn = weight_turnover(prev_lr_weight_w, lr_weight_w) if do_rebalance else 0.0
        hgb_cons_turn = weight_turnover(prev_hgb_cons_w, hgb_cons_w) if do_rebalance else 0.0
        lr_cons_turn = weight_turnover(prev_lr_cons_w, lr_cons_w) if do_rebalance else 0.0

        rows.append(
            {
                "trade_date": dt,
                "base_ret": row["net_ret"],
                "hgb_rank_ret": hgb_rank_ret - hgb_rank_turn * fee_rate,
                "lr_rank_ret": lr_rank_ret - lr_rank_turn * fee_rate,
                "hgb_weight_ret": hgb_weight_ret - hgb_weight_turn * fee_rate,
                "lr_weight_ret": lr_weight_ret - lr_weight_turn * fee_rate,
                "hgb_cons_ret": hgb_cons_ret - hgb_cons_turn * fee_rate,
                "lr_cons_ret": lr_cons_ret - lr_cons_turn * fee_rate,
                "hgb_rank_count": len(hgb_rank_w),
                "lr_rank_count": len(lr_rank_w),
                "hgb_weight_count": len(hgb_weight_w),
                "lr_weight_count": len(lr_weight_w),
                "hgb_cons_count": len(hgb_cons_w),
                "lr_cons_count": len(lr_cons_w),
                "hgb_cons_exposure": hgb_cons_expo,
                "lr_cons_exposure": lr_cons_expo,
                "rebalance_flag": int(do_rebalance),
            }
        )

        prev_hgb_rank_w = hgb_rank_w
        prev_lr_rank_w = lr_rank_w
        prev_hgb_weight_w = hgb_weight_w
        prev_lr_weight_w = lr_weight_w
        prev_hgb_cons_w = hgb_cons_w
        prev_lr_cons_w = lr_cons_w

    return pd.DataFrame(rows)


def summarize(returns: pd.Series) -> Dict[str, float]:
    returns = returns.fillna(0.0)
    return {
        "cum_return": float((1.0 + returns).prod() - 1.0),
        "avg_daily_ret": float(returns.mean()),
        "vol_daily": float(returns.std()),
        "win_rate": float((returns > 0).mean()),
    }


def plot_curve(combo: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, label in [
        ("base_ret", "Industry Top5"),
        ("hgb_rank_ret", "Top3 by ETF Trend HistGB"),
        ("lr_rank_ret", "Top3 by ETF Trend LogReg"),
        ("hgb_weight_ret", "Weighted by ETF Trend HistGB"),
        ("lr_weight_ret", "Weighted by ETF Trend LogReg"),
        ("hgb_cons_ret", "Constrained HistGB"),
        ("lr_cons_ret", "Constrained LogReg"),
    ]:
        (1.0 + combo.set_index("trade_date")[col].fillna(0.0)).cumprod().plot(ax=ax, label=label)
    ax.set_title("Industry Rotation Top5 with ETF Trend Re-Ranking")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="行业轮动 Top5 + ETF短期趋势概率模型")
    parser.add_argument("--base-dir", default="publish_repo/results/final/industry_theme_latest")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--prob-threshold", type=float, default=0.53)
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--gross-exposure", type=float, default=0.80)
    parser.add_argument("--rebalance-freq", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--output-dir", default="artifacts/combined_etf_trend_model")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_strategy = pd.read_csv(base_dir / "strategy_returns.csv")
    base_strategy["trade_date"] = pd.to_datetime(base_strategy["trade_date"])
    etf_daily = pd.read_csv(base_dir / "etf_daily.csv")

    feature_df = build_features(etf_daily)
    feature_df.to_csv(output_dir / "feature_panel.csv", index=False)

    signals = score_assets(feature_df, args.train_end)
    signals.to_csv(output_dir / "asset_signals.csv", index=False)

    combo = build_combo(
        base_strategy,
        signals,
        fee_bps=args.fee_bps,
        top_n=args.top_n,
        prob_threshold=args.prob_threshold,
        max_weight=args.max_weight,
        gross_exposure=args.gross_exposure,
        rebalance_freq=args.rebalance_freq,
    )
    combo.to_csv(output_dir / "combined_strategy.csv", index=False)

    summary = {
        "base": summarize(combo["base_ret"]),
        "histgb_rank_topn": summarize(combo["hgb_rank_ret"]),
        "logreg_rank_topn": summarize(combo["lr_rank_ret"]),
        "histgb_weighted": summarize(combo["hgb_weight_ret"]),
        "logreg_weighted": summarize(combo["lr_weight_ret"]),
        "histgb_constrained": summarize(combo["hgb_cons_ret"]),
        "logreg_constrained": summarize(combo["lr_cons_ret"]),
        "avg_hgb_rank_count": float(combo["hgb_rank_count"].mean()),
        "avg_lr_rank_count": float(combo["lr_rank_count"].mean()),
        "avg_hgb_weight_count": float(combo["hgb_weight_count"].mean()),
        "avg_lr_weight_count": float(combo["lr_weight_count"].mean()),
        "avg_hgb_cons_count": float(combo["hgb_cons_count"].mean()),
        "avg_lr_cons_count": float(combo["lr_cons_count"].mean()),
        "avg_hgb_cons_exposure": float(combo["hgb_cons_exposure"].mean()),
        "avg_lr_cons_exposure": float(combo["lr_cons_exposure"].mean()),
        "signal_asset_count": int(signals["ts_code"].nunique()),
        "signal_days": int(signals["trade_date"].nunique()),
        "constraint_config": {
            "prob_threshold": args.prob_threshold,
            "max_weight": args.max_weight,
            "gross_exposure": args.gross_exposure,
        },
        "rebalance_freq": args.rebalance_freq,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_curve(combo, output_dir / "equity_curve.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
