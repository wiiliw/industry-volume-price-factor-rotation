# -*- coding: utf-8 -*-
"""Step 4 第一轮:景气度得分卡 v1(19 席全截面)+ 滚动快照积累机制。

维度(文章四维的可用子集):
  ① 相对强度  = 60日收益 - 沪深300 60日收益(复权价,已有数据)
  ② 盈利预期  = 前十大重仓股 2027E/2026E 预期EPS增速中位数(同花顺接口,当日截面)
  ③ 估值位置  = 乐咕指数PE近1年分位(宽基)——行业级待映射表,本轮缺失记 NaN
  (成交拥挤度已被我们实验证伪,剔除)
合成:各维截面百分位(0~1),估值反向(低分位=高分),可维度平均。
快照按日期存档并追加到时序 CSV,为将来的 IC 检验积累数据。
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(k, None)

import akshare as ak
import numpy as np
import pandas as pd

REPO = Path("/Users/chenzejin/Desktop/2026实习以及春招/tmp/repo/industry-volume-price-factor-rotation")
ART = REPO / "artifacts/tushare_repro"
SNAP_DIR = REPO / "results/final/snapshots"
SNAP_DIR.mkdir(parents=True, exist_ok=True)
today = pd.Timestamp.today().normalize()

sys.path.insert(0, str(REPO))
from scr.data_service import get_tushare_pro as _gtp  # noqa: E402
pro = _gtp()

u = pd.read_csv(ART / "universe.csv").set_index("ts_code")
ind_codes = [c for c in u.index if u.loc[c, "sector_key"] != "宽基"]
name_map = dict(zip(u.index, u["name"]))
sector_map = dict(zip(u.index, u["sector_key"]))

# ---------- ① 相对强度 ----------
close_w = pd.read_csv(ART / "etf_daily.csv", parse_dates=["trade_date"]).pivot(
    index="trade_date", columns="ts_code", values="close")
bench = pd.read_csv(ART / "benchmark_daily.csv", parse_dates=["trade_date"]).set_index("trade_date").sort_index()
ret60_etf = close_w[ind_codes].pct_change(60).iloc[-1]
ret60_bench = bench["close"].pct_change(60).iloc[-1]
rel_strength = (ret60_etf - ret60_bench).dropna()

# ---------- ② 盈利预期(全部 19 席) ----------
print("拉取 19 席盈利预期(前十大重仓股 × 同花顺预测)...", flush=True)
earn_rows = {}
for etf in ind_codes:
    code6 = etf.split(".")[0]
    try:
        cons = ak.fund_portfolio_hold_em(symbol=code6, date="2026")
        stocks = cons[["股票代码"]].head(10).drop_duplicates("股票代码")["股票代码"].tolist()
        gs = []
        for sc in stocks:
            try:
                fc = ak.stock_profit_forecast_ths(symbol=sc).set_index("年度")
                e26 = float(fc.loc["2026", "均值"]) if "2026" in fc.index else np.nan
                e27 = float(fc.loc["2027", "均值"]) if "2027" in fc.index else np.nan
                g = (e27 / e26 - 1) if np.isfinite(e26) and e26 > 0 and np.isfinite(e27) else np.nan
                if np.isfinite(g):
                    gs.append(g)
                time.sleep(0.35)
            except Exception:
                continue
        earn_rows[etf] = (np.median(gs) if gs else np.nan, len(gs))
    except Exception as exc:
        earn_rows[etf] = (np.nan, 0)
        print(f"[{etf}] 失败 {str(exc)[:40]}", flush=True)

# ---------- ③ 估值位置(乐咕宽基) ----------
pe_map = {}
for sym in ["上证50", "沪深300", "中证500", "中证1000"]:
    try:
        df = ak.stock_index_pe_lg(symbol=sym)
        col = "滚动市盈率" if "滚动市盈率" in df.columns else df.columns[-2]
        pe = pd.to_numeric(df[col], errors="coerce").dropna()
        pe_map[sym] = (float(pe.iloc[-1]), (pe < pe.iloc[-1]).mean())
    except Exception:
        pass

# ---------- 合成得分卡 ----------
def pct(series_vals, val):
    s = pd.Series(series_vals).dropna()
    return (s < val).mean() if len(s) else np.nan

rows = []
for etf in ind_codes:
    rs = rel_strength.get(etf, np.nan)
    eg, nsamp = earn_rows.get(etf, (np.nan, 0))
    d1 = pct(rel_strength.values, rs) if np.isfinite(rs) else np.nan
    earn_vals = [v[0] for k, v in earn_rows.items() if np.isfinite(v[0])]
    d2 = pct(earn_vals, eg) if np.isfinite(eg) else np.nan
    # 估值:宽基席才有(本轮缺失记 NaN,下轮补映射表)
    d3 = np.nan
    dims = [x for x in (d1, d2, d3) if np.isfinite(x)]
    score = float(np.mean(dims)) if dims else np.nan
    rows.append({
        "ts_code": etf, "行业席": sector_map[etf], "名称": name_map[etf],
        "相对强度60日": round(rs, 4) if np.isfinite(rs) else np.nan,
        "相对强度分位": round(d1, 3) if np.isfinite(d1) else np.nan,
        "盈利预期增速中位数": round(eg, 4) if np.isfinite(eg) else np.nan,
        "盈利预期分位": round(d2, 3) if np.isfinite(d2) else np.nan,
        "估值分位": np.nan, "估值分位分位": np.nan,
        "景气度得分(可维度均值)": round(score, 3) if np.isfinite(score) else np.nan,
        "盈利预期样本股数": nsamp,
    })
card = pd.DataFrame(rows).sort_values("景气度得分(可维度均值)", ascending=False, na_position="last").reset_index(drop=True)
card.insert(0, "rank", card.index + 1)

print("\n===== 景气度得分卡 v1(快照 %s) =====" % today.date())
print(card.to_string(index=False))

# ---------- 快照积累 ----------
stamp = today.strftime("%Y-%m-%d")
card.to_csv(SNAP_DIR / f"scorecard_{stamp}.csv", index=False)
ts_path = REPO / "results/final/scorecard_timeseries.csv"
card["snapshot_date"] = stamp
if ts_path.exists():
    old = pd.read_csv(ts_path)
    old = old[old["snapshot_date"] != stamp]           # 当日重跑覆盖
    card = pd.concat([old, card], ignore_index=True)
card.to_csv(ts_path, index=False)
print(f"\n[快照] {SNAP_DIR / f'scorecard_{stamp}.csv'}")
print(f"[时序] {ts_path}({card['snapshot_date'].nunique()} 期)")
