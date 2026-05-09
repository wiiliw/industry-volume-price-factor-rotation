# 行业有效量价因子与行业轮动策略

这个目录是基于 Tushare 数据源的可运行复刻版，整理后可直接上传 GitHub。

## 目录结构

- `reproduce_tushare.py`
  - 主复刻脚本
- `scr/`
  - 因子计算、数据服务与辅助模块
- `docs/`
  - 原始研报、原 notebook、复刻说明
- `results/final/`
  - 最终结果
- `results/experiments/`
  - 中间实验和烟雾测试结果
- `cache/`
  - 本地抓数缓存，不建议上传

## 运行方式

安装依赖后执行：

```bash
python reproduce_tushare.py
```

纯一级行业 ETF 版本：

```bash
python reproduce_tushare.py --pure-industry-only --output-dir results/final/pure_industry
```

## 当前保留的结果

- `results/final/industry_theme_latest`
  - 行业/主题 ETF 版本，更新到 `2026-05-08`

## 可选的进一步收紧口径

当前上传目录只保留行业/主题 ETF 版本。

如果需要更接近“纯一级行业轮动”的口径，可以在本地用下面的方式再跑一版：

```bash
python reproduce_tushare.py --pure-industry-only --output-dir results/final/pure_industry
```

## 上传 GitHub 建议

- 上传源码、文档、最终结果
- 忽略 `cache/`、`__pycache__/`、`.DS_Store`
- 当前目录已经去掉纯一级行业 ETF 结果，体积更适合公开仓库
