from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import tushare as ts
from tushare.pro import client as _ts_client

DEFAULT_PROXY_URL = "https://tu.brze.top"
DEFAULT_TOKEN = "iTF8_cT707Nv8ggAO-AklOvzyVmzv-9qMu2Ao7q_cbc"
_NO_PROXY_SESSION = requests.Session()
_NO_PROXY_SESSION.trust_env = False

INCLUDE_PATTERN = (
    r"行业|证券|消费|医药|银行|军工|地产|家电|煤炭|有色|钢铁|电子|通信|传媒|"
    r"计算机|半导体|芯片|光伏|新能源|化工|汽车|农业|畜牧|建材|环保|食品|"
    r"酒|创新药|中药|5G|TMT|信息技术|材料|能源|基建|机械|机器人"
)
EXCLUDE_PATTERN = (
    r"沪深300|中证500|中证1000|上证50|创业板|科创50|A100|100ETF|300ETF|"
    r"500ETF|红利|价值|基本面|MSCI|港股|纳指|标普|德国|法国|日本|恒生|"
    r"央企|国企|综指|成指"
)
PURE_INDUSTRY_PATTERNS = {
    "consumer_staples": r"主要消费|消费80|消费$|消费指数",
    "consumer_discretionary": r"可选消费",
    "healthcare": r"医药卫生|医药健康|医药$",
    "energy": r"能源指数|全指能源",
    "information_technology": r"全指信息技术|信息技术指数",
    "materials": r"原材料|有色金属",
    "real_estate": r"房地产",
    "communication": r"通信设备|通信指数",
    "media": r"传媒指数",
    "banks": r"银行指数",
    "brokers": r"证券公司|券商",
}
PURE_INDUSTRY_EXCLUDE = (
    r"细分|主题|龙头|5G|半导体|芯片|军工龙头|环保|酒|生物医药|"
    r"计算机主题|军工指数|中证消费50|中证酒|地产主题|TMT50|科技传媒通信150|金融地产"
)

# 行业大类建池:每个行业只保留上市最早的一只(覆盖基本行业 + 一行业一席)
# 顺序即优先级:靠前的关键词先匹配(如"军工"先于"交通运输"吃掉"航天航空")
# 注意:只用基金名称匹配,基准串可能混入"XX银行活期存款利率"等公式文本造成误分
SECTOR_PATTERNS = [
    ("券商", r"证券公司|券商|证券龙头|证券行业|证券保险|上证证券|保险"),
    ("银行", r"银行"),
    ("金融地产", r"金融地产|金融"),
    ("大科技TMT", r"半导体|芯片|电子|计算机|通信|5G|TMT|信息技术|传媒|软件|光电|信息安全|云计算|人工智能"),
    ("军工", r"军工|航天|航空|国防"),
    ("医药", r"医药|医疗|中药|生物|创新药|健康"),
    ("消费", r"消费|食品|饮料|酒|白酒|零售"),
    ("家电家居", r"家电|家居"),
    ("新能源车", r"新能源车|新能源汽车|电池|锂电|智能电动汽车|储能"),
    ("光伏新能源", r"光伏|风电|绿色能源|新能源"),
    ("能源", r"能源|煤炭|石油|油气"),
    ("原材料", r"原材料|有色金属|有色|钢铁|化工|新材料|稀土|矿业"),
    ("汽车", r"汽车|零部件"),
    ("农业", r"农业|畜牧|养殖|种植|渔业"),
    ("机械", r"机械|工业母机|机床"),
    ("基建建材", r"基建|建筑材料|建材|建筑"),
    ("房地产", r"房地产|地产"),
    ("环保", r"环保"),
    ("交通运输", r"交通运输|运输|物流|机场|航运"),
]
# 宽基候选池:主流宽基指数各留最早上市一只,共同竞争"宽基"席(用户规则:宽基可入选)
BROAD_INDEX_PATTERN = (
    r"沪深300|中证500|中证1000|中证800|上证50|创业板|科创50|科创创业|"
    r"深证成指|成份|A500|北证50"
)
# 宽基反模式:名字里带这些词的是行业/SmartBeta,不是纯宽基
BROAD_NAME_ANTI_PATTERN = (
    r"医药|卫生|金融|地产|银行|非银|证券|保险|信息技术|材料|能源|汽车|"
    r"红利|价值|成长|质量|低波|ESG|现金流|动量|科技|人工智能|新能源|软件|"
    r"等权|中性|双城|G60|龙头|增强|大盘|中盘|小盘|自由"
)


def _normalize_benchmark_key(value: str) -> str:
    """归一化基准指数名,让'同一条指数'的ETF合并到同一组。

    例:中证军工指数收益率 → 中证军工指数
        中证银行指数收益率×100% → 中证银行指数
    """
    text = str(value)
    text = text.replace("收益率", "")
    text = re.sub(r"[×x][0-9.]+%", "", text)
    return text.strip()


def get_tushare_pro(
    token: Optional[str] = None,
    proxy_url: str = DEFAULT_PROXY_URL,
    timeout: int = 30,
):
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        os.environ.pop(key, None)
    requests.sessions.Session.trust_env = False
    _ts_client.requests.sessions.Session.trust_env = False
    _ts_client.requests.post = _NO_PROXY_SESSION.post
    _ts_client.DataApi._DataApi__http_url = proxy_url
    return ts.pro_api(token or DEFAULT_TOKEN, timeout=timeout)


def _ensure_ymd(date_value: str) -> str:
    return pd.Timestamp(date_value).strftime("%Y%m%d")


def _query_with_retry(pro, api_name: str, pause: float = 0.65, retries: int = 3, **kwargs):
    last_error = None
    for attempt in range(retries):
        try:
            df = getattr(pro, api_name)(**kwargs)
            time.sleep(pause)
            if df is not None:
                return df
        except Exception as exc:
            last_error = exc
            time.sleep(pause * (attempt + 1))
    if last_error is not None:
        raise last_error
    return pd.DataFrame()


def get_trade_calendar(pro, start_date: str, end_date: str) -> pd.DataFrame:
    return _query_with_retry(
        pro,
        "trade_cal",
        pause=0.2,
        retries=5,
        exchange="SSE",
        start_date=_ensure_ymd(start_date),
        end_date=_ensure_ymd(end_date),
        fields="cal_date,is_open",
    )


def get_industry_etf_universe(
    pro,
    end_date: str,
    min_list_date: str = "20100101",
    max_symbols: Optional[int] = None,
    strict_mode: bool = False,
    fit_end_date: Optional[str] = None,
    pure_industry_only: bool = False,
    max_per_sector: Optional[int] = 1,
) -> pd.DataFrame:
    fund_basic = _query_with_retry(
        pro,
        "fund_basic",
        pause=0.5,
        retries=5,
        market="E",
        status="L",
        fields="ts_code,name,fund_type,invest_type,list_date,benchmark",
    )

    mask = (
        fund_basic["fund_type"].eq("股票型")
        & fund_basic["invest_type"].fillna("").str.contains("被动指数")
        & fund_basic["name"].fillna("").str.contains("ETF")
    )
    df = fund_basic.loc[mask].copy()

    # ---- 宽基候选池:主流宽基指数各留最早上市一只(独立于行业筛选) ----
    broad_src = fund_basic["name"].fillna("")
    broad_mask = (
        fund_basic["fund_type"].eq("股票型")
        & fund_basic["invest_type"].fillna("").str.contains("被动指数")
        & broad_src.str.contains("ETF")
        & broad_src.str.contains(BROAD_INDEX_PATTERN, regex=True)
        & ~broad_src.str.contains(BROAD_NAME_ANTI_PATTERN, regex=True)
        & ~broad_src.str.contains(r"联接|LOF|港股|香港|恒生|纳指|标普|QDII", regex=True)
    )
    broad = fund_basic.loc[broad_mask].copy()
    broad = broad[
        broad["list_date"].between(_ensure_ymd(min_list_date), _ensure_ymd(end_date))
    ]
    broad["benchmark_key"] = broad["benchmark"].fillna(broad["name"]).map(_normalize_benchmark_key)
    broad = (
        broad.sort_values(["benchmark_key", "list_date", "ts_code"])
        .groupby("benchmark_key", as_index=False)
        .first()
        .sort_values(["list_date", "ts_code"])
        .reset_index(drop=True)
    )
    broad["sector_key"] = "宽基"

    include_mask = df["name"].str.contains(INCLUDE_PATTERN, regex=True) | df[
        "benchmark"
    ].fillna("").str.contains(INCLUDE_PATTERN, regex=True)
    exclude_mask = df["name"].str.contains(EXCLUDE_PATTERN, regex=True) | df[
        "benchmark"
    ].fillna("").str.contains(EXCLUDE_PATTERN, regex=True)

    df = df.loc[include_mask & ~exclude_mask].copy()
    # 排除场外联接/LOF(如"XX ETF联接(LOF)-A"):非场内行业ETF工具;宽基按用户规则可入选
    feeder_mask = df["name"].str.contains(r"联接|LOF", regex=True)
    df = df.loc[~feeder_mask].copy()
    df = df[df["list_date"].between(_ensure_ymd(min_list_date), _ensure_ymd(end_date))]
    df["benchmark_key"] = df["benchmark"].fillna(df["name"]).map(_normalize_benchmark_key)
    df = (
        df.sort_values(["benchmark_key", "list_date", "ts_code"])
        .groupby("benchmark_key", as_index=False)
        .first()
        .sort_values(["list_date", "ts_code"])
        .reset_index(drop=True)
    )

    if strict_mode:
        strict_exclude = (
            r"港股|香港|恒生|纳指|标普|德国|法国|日本|东南亚|沪港深|中韩|"
            r"科创|创业板|跨境|海外|AH|红利|央企|国企|卫星|线上消费|"
            r"在线消费|碳中和|品牌消费|消费服务|创新药|机器人"
        )
        strict_name_exclude = df["name"].str.contains(strict_exclude, regex=True)
        strict_bench_exclude = df["benchmark"].fillna("").str.contains(
            strict_exclude, regex=True
        )
        df = df.loc[~(strict_name_exclude | strict_bench_exclude)].copy()
        # 用户要求覆盖基本行业:不再按训练期截断上市时间,保留全部严格筛选后的行业ETF
        # (晚期上市ETF在训练窗口自然缺特征/标签,不影响模型;预测期自动纳入)

    # 行业大类建池:按 max_per_sector 决定每行业保留几只(None=全部,扩大排序截面)
    if not pure_industry_only:
        source = df["name"].fillna("")  # 只按基金名称分类,避免基准串公式文本干扰
        sector_keys = pd.Series(index=df.index, dtype="object")
        for sector_name, pattern in SECTOR_PATTERNS:
            # 首个匹配生效:仅对尚未分类的成员赋值(mask 语义是覆盖,需配合 isna 条件)
            cond = source.str.contains(pattern, regex=True) & sector_keys.isna()
            sector_keys = sector_keys.mask(cond, sector_name)
        df = df.loc[sector_keys.notna()].copy()
        df["sector_key"] = sector_keys.loc[df.index].values
        if max_per_sector is not None:
            df = (
                df.sort_values(["sector_key", "list_date", "ts_code"])
                .groupby("sector_key", as_index=False)
                .head(max_per_sector)
                .sort_values(["list_date", "ts_code"])
                .reset_index(drop=True)
            )

    if pure_industry_only:
        base = pd.Series("", index=df.index)
        source = df["benchmark"].fillna("") + " " + df["name"].fillna("")
        sector_keys = pd.Series(index=df.index, dtype="object")
        for sector_key, pattern in PURE_INDUSTRY_PATTERNS.items():
            sector_keys = sector_keys.mask(source.str.contains(pattern, regex=True), sector_key)
        exclude_mask = source.str.contains(PURE_INDUSTRY_EXCLUDE, regex=True)
        df = df.loc[sector_keys.notna() & ~exclude_mask].copy()
        df["sector_key"] = sector_keys.loc[df.index].values
        df = (
            df.sort_values(["sector_key", "list_date", "ts_code"])
            .groupby("sector_key", as_index=False)
            .first()
            .sort_values(["list_date", "ts_code"])
            .reset_index(drop=True)
        )

    # 合并:行业池 + 宽基候选池(行业池成员的 sector_key 已在上方赋值)
    universe = pd.concat([df, broad], ignore_index=True)
    universe = (
        universe.drop_duplicates(subset="ts_code")
        .sort_values(["sector_key", "list_date", "ts_code"])
        .reset_index(drop=True)
    )

    if max_symbols is not None:
        universe = universe.head(max_symbols).copy()

    cols = ["ts_code", "name", "fund_type", "invest_type", "list_date", "benchmark"]
    if "sector_key" in universe.columns:
        cols.append("sector_key")
    return universe[cols].reset_index(drop=True)


def resolve_latest_end_date(
    pro,
    today: str,
    probe_code: str = "510300.SH",
) -> str:
    calendar = get_trade_calendar(
        pro,
        start_date=(pd.Timestamp(today) - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        end_date=today,
    )
    open_dates = (
        pd.to_datetime(calendar.loc[calendar["is_open"] == 1, "cal_date"])
        .sort_values(ascending=False)
        .dt.strftime("%Y%m%d")
        .tolist()
    )
    for ymd in open_dates:
        df = _query_with_retry(
            pro,
            "fund_daily",
            pause=0.2,
            retries=5,
            ts_code=probe_code,
            trade_date=ymd,
            fields="ts_code,trade_date,close",
        )
        if df is not None and not df.empty:
            return pd.Timestamp(ymd).strftime("%Y-%m-%d")
    raise RuntimeError(f"未能解析 {today} 之前的最新可用交易日")


def _standardize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"vol": "volume"}
    out = df.rename(columns=rename_map).copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out = out.sort_values("trade_date").reset_index(drop=True)
    out["factor"] = 1.0
    return out


def get_ts_etf_price(
    pro,
    ts_code: str,
    start_date: str,
    end_date: str,
    pause: float = 0.65,
) -> pd.DataFrame:
    df = _query_with_retry(
        pro,
        "fund_daily",
        pause=pause,
        ts_code=ts_code,
        start_date=_ensure_ymd(start_date),
        end_date=_ensure_ymd(end_date),
        fields="ts_code,trade_date,open,high,low,close,vol,amount",
    )
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "factor",
            ]
        )
    # 复权处理:份额折算/分红造成假跳变,用 fund_adj 因子前复权(相对最新因子)
    adj = _query_with_retry(
        pro,
        "fund_adj",
        pause=pause,
        ts_code=ts_code,
        start_date=_ensure_ymd(start_date),
        end_date=_ensure_ymd(end_date),
        fields="trade_date,adj_factor",
    )
    if adj is not None and not adj.empty:
        adj["trade_date"] = pd.to_datetime(adj["trade_date"])
        adj = adj.sort_values("trade_date").drop_duplicates("trade_date")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.merge(adj, on="trade_date", how="left")
        df["adj_factor"] = df["adj_factor"].ffill().bfill()
        latest = float(df["adj_factor"].iloc[-1])
        if latest > 0:
            scale = df["adj_factor"] / latest
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col] * scale
        df = df.drop(columns=["adj_factor"])
    return _standardize_price_frame(df)


def get_ts_index_price(
    pro,
    ts_code: str,
    start_date: str,
    end_date: str,
    pause: float = 0.65,
) -> pd.DataFrame:
    query_args = dict(
        start_date=_ensure_ymd(start_date),
        end_date=_ensure_ymd(end_date),
        fields="ts_code,trade_date,open,high,low,close,vol,amount",
    )
    df = _query_with_retry(pro, "index_daily", pause=pause, ts_code=ts_code, **query_args)
    if (df is None or df.empty) and ts_code == "399300.SZ":
        df = _query_with_retry(pro, "index_daily", pause=pause, ts_code="000300.SH", **query_args)
    if (df is None or df.empty) and ts_code == "000300.SH":
        df = _query_with_retry(pro, "index_daily", pause=pause, ts_code="399300.SZ", **query_args)
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "factor",
            ]
        )
    return _standardize_price_frame(df)


def _cache_path(cache_dir: Path, ts_code: str) -> Path:
    safe_code = re.sub(r"[^A-Za-z0-9_.-]+", "_", ts_code)
    return cache_dir / f"{safe_code}.csv"


def load_cached_prices(
    cache_dir: Path,
    ts_code: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    path = _cache_path(cache_dir, ts_code)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["trade_date"])
    if df.empty:
        return None
    min_dt = df["trade_date"].min()
    max_dt = df["trade_date"].max()
    if min_dt <= pd.Timestamp(start_date) and max_dt >= pd.Timestamp(end_date):
        return df[
            (df["trade_date"] >= pd.Timestamp(start_date))
            & (df["trade_date"] <= pd.Timestamp(end_date))
        ].copy()
    return None


def save_cached_prices(cache_dir: Path, ts_code: str, df: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(_cache_path(cache_dir, ts_code), index=False)


def get_many_etf_prices(
    pro,
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    pause: float = 0.65,
) -> pd.DataFrame:
    frames = []
    for _, row in universe.iterrows():
        ts_code = row["ts_code"]
        cached = load_cached_prices(cache_dir, ts_code, start_date, end_date)
        if cached is not None:
            frames.append(cached)
            continue
        df = get_ts_etf_price(pro, ts_code, start_date, end_date, pause=pause)
        if not df.empty:
            save_cached_prices(cache_dir, ts_code, df)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def panelize_price_data(price_df: pd.DataFrame) -> pd.DataFrame:
    fields = ["open", "high", "low", "close", "volume", "amount", "factor"]
    panels = []
    for field in fields:
        wide = price_df.pivot(index="trade_date", columns="ts_code", values=field)
        wide.columns = pd.MultiIndex.from_product([[field], wide.columns])
        panels.append(wide)
        if field == "volume":
            alias = wide.copy()
            alias.columns = pd.MultiIndex.from_product([["vol"], alias.columns.get_level_values(1)])
            panels.append(alias)
    data = pd.concat(panels, axis=1).sort_index(axis=1)
    data.index = pd.to_datetime(data.index)
    return data.sort_index()
