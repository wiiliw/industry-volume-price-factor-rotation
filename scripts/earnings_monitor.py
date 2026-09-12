# -*- coding: utf-8 -*-
"""盈利预期监控 v1:对当前持仓的每只行业 ETF,取前十大重仓股作为行业样本,
用同花顺盈利预测接口计算"预期EPS增速(2027E/2026E)"与"EPS相对行业平均",
输出 results/final/earnings_forecast_monitor.csv。
局限:样本=前十大重仓股(非全成分);2026E 为预测值(机构一致预期口径)。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(k, None)

import akshare as ak
import numpy as np
import pandas as pd

REPO = Path("/Users/chenzejin/Desktop/2026实习以及春招/tmp/repo/industry-volume-price-factor-rotation")
ART = REPO / "artifacts/tushare_repro"
OUT = REPO / "results/final/earnings_forecast_monitor.csv"

rot = pd.read_csv(ART / "strategy_returns.csv", parse_dates=["trade_date"]).set_index("trade_date")
u = pd.read_csv(ART / "universe.csv")
name_map = dict(zip(u.ts_code, u["name"]))
sector_map = dict(zip(u.ts_code, u["sector_key"]))

d_sig = rot.index[rot["rebalanced"] == 1][-1]
hold = [h for h in str(rot.loc[d_sig, "holdings"]).split(",") if h]
w = [float(x) for x in str(rot.loc[d_sig, "weights"]).split(",")]
wmap = dict(zip(hold, w))

rows = []
for etf in hold:
    code6 = etf.split(".")[0]
    try:
        cons = ak.fund_portfolio_hold_em(symbol=code6, date="2026")
    except Exception as exc:
        print(f"[{etf}] 重仓股获取失败: {exc}", flush=True)
        continue
    stocks = cons[["股票代码", "股票名称"]].head(10).drop_duplicates("股票代码")
    per = []
    for _, s in stocks.iterrows():
        sc, sn = s["股票代码"], s["股票名称"]
        try:
            fc = ak.stock_profit_forecast_ths(symbol=sc)
            fc = fc.set_index("年度")
            e26 = float(fc.loc["2026", "均值"]) if "2026" in fc.index else np.nan
            e27 = float(fc.loc["2027", "均值"]) if "2027" in fc.index else np.nan
            ind_avg = float(fc.loc["2026", "行业平均数"]) if "2026" in fc.index else np.nan
            growth = (e27 / e26 - 1) if np.isfinite(e26) and e26 > 0 and np.isfinite(e27) else np.nan
            per.append({"股票": f"{sn}({sc})", "eps26": e26, "eps27": e27,
                        "growth27": growth, "ind_avg": ind_avg})
            time.sleep(0.4)
        except Exception:
            continue
    if not per:
        continue
    pdf = pd.DataFrame(per)
    g_med = float(pdf["growth27"].median()) if pdf["growth27"].notna().any() else np.nan
    rel = float((pdf["eps26"] / pdf["ind_avg"].replace(0, np.nan)).median())
    rows.append({
        "ETF": etf, "行业席": sector_map.get(etf, "?"), "ETF名称": name_map.get(etf, etf),
        "权重": f"{wmap[etf]:.1%}", "样本股数": len(pdf),
        "盈利预期增速(2027E/2026E中位数)": round(g_med, 4) if np.isfinite(g_med) else np.nan,
        "EPS相对行业平均(中位数)": round(rel, 3) if np.isfinite(rel) else np.nan,
        "明细": "; ".join(f"{r['股票']}:{r['growth27']:+.0%}" if np.isfinite(r["growth27"]) else f"{r['股票']}:NA"
                          for _, r in pdf.iterrows()),
    })
    print(f"[{etf}] 完成", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print("\n===== 盈利预期监控 v1 =====")
print(res[["ETF", "行业席", "权重", "样本股数", "盈利预期增速(2027E/2026E中位数)", "EPS相对行业平均(中位数)"]].to_string(index=False))
print(f"\n[输出] {OUT}")
