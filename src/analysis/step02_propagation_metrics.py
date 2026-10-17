# -*- coding: utf-8 -*-
"""
==============================================================================
研究项目：标题和封面双模态协同对社交媒体平台视频传播的影响
            ——以 B 站知识类视频为例
步骤二：传播效果指标构建 (Propagation Effect Metrics)
==============================================================================

研究单位：单个视频 (one row = one video)

本脚本基于上一步产物 outputs/data_clean_base.csv，
构建“因变量体系”（主因变量 / 辅助因变量 / 稳健性因变量 / 极端值标记），
并输出处理后的数据、描述统计、分布图、变量说明表与质量检查报告。

设计原则（严格遵循需求）：
  1. 绝不覆盖原始 data_clean_filtered.csv，也不覆盖输入 data_clean_base.csv。
  2. 所有产物写入 outputs/ 下的新文件。
  3. 缺失值 / 异常值 / 模型不可用 / 数据问题不静默跳过，全部写日志。
  4. 本步骤只“标记”极端值，不删除。
  5. 变量名用英文，论文解释用中文别名（见 variable_dictionary_step2.xlsx）。

依赖包（建议在干净虚拟环境中安装）：
  pandas>=2.0  numpy>=1.24  scipy>=1.10  matplotlib>=3.7  openpyxl>=3.1
安装：
  pip install pandas numpy scipy matplotlib openpyxl
运行：
  python step2_propagation_metrics.py
==============================================================================
"""

from __future__ import annotations
import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")  # 无界面后端，服务器/批处理可用
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
# 0. 路径配置（可按你的项目结构修改）
# ------------------------------------------------------------------ #
INPUT_PATH      = "outputs/data_clean_base.csv"     # 步骤一产物（只读，不修改）
OUTPUT_DIR      = "outputs"
FIG_DIR         = os.path.join(OUTPUT_DIR, "figures")
LOG_DIR         = os.path.join(OUTPUT_DIR, "logs")
DESC_DIR        = os.path.join(OUTPUT_DIR, "descriptive")

OUT_METRICS_CSV = os.path.join(OUTPUT_DIR, "propagation_metrics.csv")
OUT_SUMMARY_XLSX= os.path.join(DESC_DIR,   "propagation_metric_summary.xlsx")
OUT_DICT_XLSX   = os.path.join(OUTPUT_DIR, "variable_dictionary_step2.xlsx")
FIG_LOGVIEW     = os.path.join(FIG_DIR,    "propagation_distribution_log_view.png")
FIG_LOGVPD      = os.path.join(FIG_DIR,    "propagation_distribution_log_view_per_day.png")
FIG_RECOG       = os.path.join(FIG_DIR,    "propagation_distribution_recognition.png")
FIG_PARTI       = os.path.join(FIG_DIR,    "propagation_distribution_participation.png")

# 标准化口径：ddof=0 = 总体标准差（与 sklearn StandardScaler 一致）。如需样本标准差改为 1。
ZSCORE_DDOF = 0

RANDOM_STATE = 42  # 本步骤无随机过程，仅为占位保证可复现


# ------------------------------------------------------------------ #
# 1. 目录与日志
# ------------------------------------------------------------------ #
for d in (OUTPUT_DIR, FIG_DIR, LOG_DIR, DESC_DIR):
    os.makedirs(d, exist_ok=True)

LOG_PATH = os.path.join(LOG_DIR, "step2_propagation_metrics.log")

logger = logging.getLogger("step2")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
_fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_sh)

# 质量检查累加器：每条 (检查项, 状态, 详情)
QC_ROWS: list[dict] = []
def qc(item: str, status: str, detail: str = ""):
    QC_ROWS.append({"check_item": item, "status": status, "detail": detail})
    lvl = {"OK": logging.INFO, "WARN": logging.WARNING, "FAIL": logging.ERROR}.get(status, logging.INFO)
    logger.log(lvl, f"[QC] {item} -> {status} {detail}")


# ------------------------------------------------------------------ #
# 2. 工具函数
# ------------------------------------------------------------------ #
def log1p_safe(s: pd.Series, name: str) -> pd.Series:
    """对 ln(x+1) 做安全计算：负值会使结果为 NaN，先检测并记录。"""
    n_neg = int((s < 0).sum())
    if n_neg:
        qc(f"log1p::{name}", "WARN", f"{n_neg} 行底数为负，ln(x+1) 将产生 NaN，请检查上游清洗")
    out = np.log1p(s.astype("float64"))
    return out


def zscore(s: pd.Series, ddof: int = ZSCORE_DDOF) -> pd.Series:
    """标准化为 z 分数。std=0（常数列）时返回全 0 并记录。"""
    mu = s.mean()
    sd = s.std(ddof=ddof)
    if not np.isfinite(sd) or sd == 0:
        qc(f"zscore::{s.name}", "WARN", "标准差为 0 或非有限，z 分数置为 0")
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def rate(numer: pd.Series, denom_plus1: pd.Series, name: str) -> pd.Series:
    """互动率 = 分子 / (view+1)。检测 >1 的异常比率并记录（理论上点赞数可>播放数，仅提示）。"""
    r = numer.astype("float64") / denom_plus1
    n_gt1 = int((r > 1).sum())
    if n_gt1:
        qc(f"rate::{name}", "WARN", f"{n_gt1} 行比率 > 1（分子大于 view，可能为缓存/计数口径差异），保留不裁剪")
    return r


# ------------------------------------------------------------------ #
# 3. 读取输入并校验
# ------------------------------------------------------------------ #
logger.info("=" * 70)
logger.info("步骤二启动：传播效果指标构建")
logger.info(f"输入文件：{INPUT_PATH}")

if not os.path.exists(INPUT_PATH):
    qc("input_file", "FAIL", f"未找到 {INPUT_PATH}")
    logger.error("输入文件不存在，终止。请先完成步骤一并生成 data_clean_base.csv。")
    sys.exit(1)

df = pd.read_csv(INPUT_PATH)
N0 = len(df)
qc("input_file", "OK", f"读取成功，{N0} 行 × {df.shape[1]} 列")

# 必需的原始计数字段
RAW_COUNT_COLS = ["view", "like", "coin", "favorite", "share", "danmaku", "reply", "fans"]
missing_cols = [c for c in RAW_COUNT_COLS if c not in df.columns]
if missing_cols:
    qc("required_columns", "FAIL", f"缺少字段：{missing_cols}")
    logger.error("关键计数字段缺失，终止。")
    sys.exit(1)
qc("required_columns", "OK", f"计数字段齐全：{RAW_COUNT_COLS}")

# bvid 唯一性（研究单位是单个视频）
if "bvid" in df.columns:
    dup = int(df["bvid"].duplicated().sum())
    qc("bvid_unique", "OK" if dup == 0 else "WARN", f"重复 bvid 行数 = {dup}")

# ------------------------------------------------------------------ #
# 3b. 计数字段缺失值 / 负值检查（不静默处理）
# ------------------------------------------------------------------ #
for c in RAW_COUNT_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    n_na  = int(df[c].isna().sum())
    n_neg = int((df[c] < 0).sum())
    if n_na:
        qc(f"missing::{c}", "WARN", f"{n_na} 个缺失值，ln/比率将变 NaN（保留，不填补）")
    if n_neg:
        qc(f"negative::{c}", "WARN", f"{n_neg} 个负值（计数不应为负），请回查上游")
    if n_na == 0 and n_neg == 0:
        qc(f"clean::{c}", "OK", f"min={df[c].min()}, max={df[c].max()}")

# ------------------------------------------------------------------ #
# 3c. video_age_days：优先复用步骤一，缺失则由 pubdate 兜底重建
# ------------------------------------------------------------------ #
if "video_age_days" in df.columns and df["video_age_days"].notna().all():
    df["video_age_days"] = pd.to_numeric(df["video_age_days"], errors="coerce")
    qc("video_age_days", "OK", "复用步骤一字段")
else:
    qc("video_age_days", "WARN", "缺失或不完整，尝试用 pubdate 重建（参照系=数据内最晚 pubdate）")
    if "pubdate" not in df.columns:
        qc("video_age_days::rebuild", "FAIL", "无 pubdate，无法重建 video_age_days")
        logger.error("无法构建 view_per_day，终止。")
        sys.exit(1)
    pub = pd.to_datetime(df["pubdate"], errors="coerce")
    n_bad = int(pub.isna().sum())
    if n_bad:
        qc("pubdate_parse", "WARN", f"{n_bad} 个 pubdate 无法解析，对应 age 为 NaN")
    ref = pub.max()
    df["video_age_days"] = (ref - pub).dt.total_seconds() / 86400.0

n_age_bad = int(df["video_age_days"].isna().sum()) + int((df["video_age_days"] < 0).sum())
if n_age_bad:
    qc("video_age_days::valid", "WARN", f"{n_age_bad} 行 age 为 NaN 或 <0，view_per_day 受影响")


# ------------------------------------------------------------------ #
# 4. 构建传播效果变量
# ------------------------------------------------------------------ #
logger.info("-" * 70)
logger.info("开始构建传播效果变量")

# --- 4.1 播放规模（主因变量族） ---
df["log_view"]          = log1p_safe(df["view"], "view")                       # 1
df["view_per_day"]      = df["view"].astype("float64") / (df["video_age_days"] + 1)  # 2
df["log_view_per_day"]  = log1p_safe(df["view_per_day"], "view_per_day")       # 3

# --- 4.2 认可度 / 参与度（辅助因变量） ---
df["recognition_raw"]   = df[["like", "coin", "favorite", "share"]].sum(axis=1)  # 4
df["recognition"]       = log1p_safe(df["recognition_raw"], "recognition_raw")   # 5
df["participation_raw"] = df[["danmaku", "reply"]].sum(axis=1)                    # 6
df["participation"]     = log1p_safe(df["participation_raw"], "participation_raw")# 7

# --- 4.3 标准化 + 综合传播效果指数（稳健性 / 综合视角） ---
df["z_view"]          = zscore(df["log_view"])        # 8
df["z_recognition"]   = zscore(df["recognition"])     # 9
df["z_participation"] = zscore(df["participation"])   # 10
df["communication_effect"] = df[["z_view", "z_recognition", "z_participation"]].mean(axis=1)  # 11

# --- 4.4 相对播放（控制粉丝基数） ---
if "log_fans" in df.columns and df["log_fans"].notna().all():
    qc("log_fans", "OK", "复用步骤一字段")
else:
    qc("log_fans", "WARN", "缺失/不完整，按 ln(fans+1) 重建")
    df["log_fans"] = log1p_safe(df["fans"], "fans")
df["relative_view"] = df["log_view"] - df["log_fans"]  # 12

# --- 4.5 互动率（稳健性，规模无关的相对强度） ---
denom = df["view"].astype("float64") + 1.0
df["like_rate"]     = rate(df["like"],     denom, "like_rate")      # 13
df["coin_rate"]     = rate(df["coin"],     denom, "coin_rate")      # 14
df["favorite_rate"] = rate(df["favorite"], denom, "favorite_rate")  # 15
df["share_rate"]    = rate(df["share"],    denom, "share_rate")     # 16
df["comment_rate"]  = rate(df["reply"],    denom, "comment_rate")   # 17
df["danmaku_rate"]  = rate(df["danmaku"],  denom, "danmaku_rate")   # 18

# ------------------------------------------------------------------ #
# 5. 极端值标记（只标记，不删除）
# ------------------------------------------------------------------ #
logger.info("-" * 70)
logger.info("生成极端值 / 高传播标记（不删除任何行）")

view_p99 = df["view"].quantile(0.99)
fans_p99 = df["fans"].quantile(0.99)
lvpd_med = df["log_view_per_day"].median()
lvpd_q75 = df["log_view_per_day"].quantile(0.75)

df["top1_view"]          = (df["view"] >= view_p99).astype(int)             # 播放量前 1%
df["top1_fans"]          = (df["fans"] >= fans_p99).astype(int)             # 粉丝量前 1%
df["high_spread_median"] = (df["log_view_per_day"] > lvpd_med).astype(int)  # 高于中位数
df["high_spread_q75"]    = (df["log_view_per_day"] >= lvpd_q75).astype(int) # 前 25%

for m, thr in [("top1_view", view_p99), ("top1_fans", fans_p99),
               ("high_spread_median", lvpd_med), ("high_spread_q75", lvpd_q75)]:
    qc(f"marker::{m}", "OK", f"阈值={thr:.4f}, 命中={int(df[m].sum())} 行")


# ------------------------------------------------------------------ #
# 6. 构建后完整性校验：NaN / Inf
# ------------------------------------------------------------------ #
NEW_VARS = [
    "log_view", "view_per_day", "log_view_per_day",
    "recognition_raw", "recognition", "participation_raw", "participation",
    "z_view", "z_recognition", "z_participation", "communication_effect",
    "relative_view",
    "like_rate", "coin_rate", "favorite_rate", "share_rate", "comment_rate", "danmaku_rate",
]
for v in NEW_VARS:
    s = df[v]
    n_na  = int(s.isna().sum())
    n_inf = int(np.isinf(s.to_numpy(dtype="float64", na_value=np.nan)).sum())
    if n_na or n_inf:
        qc(f"postcheck::{v}", "WARN", f"NaN={n_na}, Inf={n_inf}")
    else:
        qc(f"postcheck::{v}", "OK", f"无 NaN/Inf；mean={s.mean():.4f}, std={s.std():.4f}")


# ------------------------------------------------------------------ #
# 7. 分布图：原始 vs 对数，标注偏态改善
# ------------------------------------------------------------------ #
logger.info("-" * 70)
logger.info("绘制分布图（原始 vs 对数）")

# 图中使用英文标签，规避容器内中文字体缺失导致的乱码（中文释义见变量字典）
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})

SKEW_ROWS: list[dict] = []
def plot_raw_vs_log(raw: pd.Series, log: pd.Series, raw_name: str, log_name: str,
                    title: str, out_path: str):
    r = raw.replace([np.inf, -np.inf], np.nan).dropna()
    l = log.replace([np.inf, -np.inf], np.nan).dropna()
    sk_r = float(stats.skew(r)) if len(r) > 2 else np.nan
    sk_l = float(stats.skew(l)) if len(l) > 2 else np.nan
    SKEW_ROWS.append({"variable_pair": f"{raw_name} -> {log_name}",
                      "skew_raw": sk_r, "skew_log": sk_l,
                      "abs_skew_reduction": abs(sk_r) - abs(sk_l)})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(r, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_title(f"{raw_name} (raw)\nskewness = {sk_r:.3f}")
    axes[0].set_xlabel(raw_name); axes[0].set_ylabel("count")
    axes[1].hist(l, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].set_title(f"{log_name} (log)\nskewness = {sk_l:.3f}")
    axes[1].set_xlabel(log_name); axes[1].set_ylabel("count")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    qc(f"figure::{os.path.basename(out_path)}", "OK",
       f"skew {sk_r:.3f} -> {sk_l:.3f}")

plot_raw_vs_log(df["view"], df["log_view"], "view", "log_view",
                "Distribution of View vs log(View+1)", FIG_LOGVIEW)
plot_raw_vs_log(df["view_per_day"], df["log_view_per_day"], "view_per_day", "log_view_per_day",
                "Distribution of View-per-day vs log(View-per-day+1)", FIG_LOGVPD)
# 额外覆盖“所有传播变量”：认可度、参与度
plot_raw_vs_log(df["recognition_raw"], df["recognition"], "recognition_raw", "recognition",
                "Distribution of Recognition vs log(Recognition+1)", FIG_RECOG)
plot_raw_vs_log(df["participation_raw"], df["participation"], "participation_raw", "participation",
                "Distribution of Participation vs log(Participation+1)", FIG_PARTI)


# ------------------------------------------------------------------ #
# 8. 描述统计汇总
# ------------------------------------------------------------------ #
logger.info("-" * 70)
logger.info("生成描述统计与偏态对比")

DESC_VARS = [
    "view", "log_view", "view_per_day", "log_view_per_day",
    "recognition_raw", "recognition", "participation_raw", "participation",
    "z_view", "z_recognition", "z_participation", "communication_effect",
    "relative_view", "log_fans",
    "like_rate", "coin_rate", "favorite_rate", "share_rate", "comment_rate", "danmaku_rate",
]
desc_rows = []
for v in DESC_VARS:
    s = df[v].replace([np.inf, -np.inf], np.nan).dropna()
    desc_rows.append({
        "variable": v, "n": int(s.size),
        "mean": s.mean(), "std": s.std(),
        "min": s.min(), "q25": s.quantile(.25), "median": s.median(),
        "q75": s.quantile(.75), "max": s.max(),
        "skew": float(stats.skew(s)) if s.size > 2 else np.nan,
        "kurtosis": float(stats.kurtosis(s)) if s.size > 2 else np.nan,
    })
desc_df = pd.DataFrame(desc_rows)
skew_df = pd.DataFrame(SKEW_ROWS)

marker_df = pd.DataFrame([
    {"marker": "top1_view",          "hits": int(df["top1_view"].sum()),          "share_%": 100*df["top1_view"].mean(),          "threshold": view_p99},
    {"marker": "top1_fans",          "hits": int(df["top1_fans"].sum()),          "share_%": 100*df["top1_fans"].mean(),          "threshold": fans_p99},
    {"marker": "high_spread_median", "hits": int(df["high_spread_median"].sum()), "share_%": 100*df["high_spread_median"].mean(), "threshold": lvpd_med},
    {"marker": "high_spread_q75",    "hits": int(df["high_spread_q75"].sum()),    "share_%": 100*df["high_spread_q75"].mean(),    "threshold": lvpd_q75},
])
qc_df = pd.DataFrame(QC_ROWS)

with pd.ExcelWriter(OUT_SUMMARY_XLSX, engine="openpyxl") as xw:
    desc_df.to_excel(xw,   sheet_name="descriptive_stats", index=False)
    skew_df.to_excel(xw,   sheet_name="skewness_comparison", index=False)
    marker_df.to_excel(xw, sheet_name="extreme_markers", index=False)
    qc_df.to_excel(xw,     sheet_name="quality_check", index=False)
qc("output::summary_xlsx", "OK", OUT_SUMMARY_XLSX)


# ------------------------------------------------------------------ #
# 9. 变量说明表（英文名 + 中文别名 + 公式 + 角色定位）
# ------------------------------------------------------------------ #
DICT_ROWS = [
    # name, cn_alias, formula, dtype, role, dv_type, note
    ("log_view", "对数播放量", "ln(view+1)", "float", "主因变量", "核心 DV",
     "传播广度的首选因变量，压缩右偏长尾"),
    ("view_per_day", "日均播放量", "view/(video_age_days+1)", "float", "中间变量", "—",
     "去除曝光时长差异，原始尺度右偏"),
    ("log_view_per_day", "对数日均播放量", "ln(view_per_day+1)", "float", "主因变量", "核心 DV",
     "控制视频年龄后的传播速度，主模型可选首选项"),
    ("recognition_raw", "认可度原始值", "like+coin+favorite+share", "int", "中间变量", "—",
     "正向互动总量（认可类）"),
    ("recognition", "对数认可度", "ln(recognition_raw+1)", "float", "辅助因变量", "辅助 DV",
     "衡量观众认可强度，用于机制/稳健性分析"),
    ("participation_raw", "参与度原始值", "danmaku+reply", "int", "中间变量", "—",
     "讨论类互动总量"),
    ("participation", "对数参与度", "ln(participation_raw+1)", "float", "辅助因变量", "辅助 DV",
     "衡量观众讨论参与，用于机制/稳健性分析"),
    ("z_view", "标准化播放", "zscore(log_view)", "float", "构造项", "—",
     "用于合成综合指数"),
    ("z_recognition", "标准化认可度", "zscore(recognition)", "float", "构造项", "—",
     "用于合成综合指数"),
    ("z_participation", "标准化参与度", "zscore(participation)", "float", "构造项", "—",
     "用于合成综合指数"),
    ("communication_effect", "综合传播效果指数", "mean(z_view,z_recognition,z_participation)", "float",
     "稳健性因变量", "稳健性 DV", "综合视角，禁止作为唯一核心 DV"),
    ("relative_view", "相对播放（去粉丝）", "log_view - log_fans", "float", "稳健性因变量", "稳健性 DV",
     "控制 UP 主粉丝基数后的相对传播力"),
    ("like_rate", "点赞率", "like/(view+1)", "float", "稳健性因变量", "稳健性 DV", "规模无关的互动强度"),
    ("coin_rate", "投币率", "coin/(view+1)", "float", "稳健性因变量", "稳健性 DV", "高成本认可强度"),
    ("favorite_rate", "收藏率", "favorite/(view+1)", "float", "稳健性因变量", "稳健性 DV", "内容留存价值"),
    ("share_rate", "分享率", "share/(view+1)", "float", "稳健性因变量", "稳健性 DV", "二次传播倾向"),
    ("comment_rate", "评论率", "reply/(view+1)", "float", "稳健性因变量", "稳健性 DV", "深度讨论强度"),
    ("danmaku_rate", "弹幕率", "danmaku/(view+1)", "float", "稳健性因变量", "稳健性 DV", "实时观看参与"),
    ("top1_view", "播放量前1%标记", "view>=P99(view)", "int(0/1)", "标记变量", "—", "极端高播放，仅标记不删除"),
    ("top1_fans", "粉丝量前1%标记", "fans>=P99(fans)", "int(0/1)", "标记变量", "—", "头部 UP 主，仅标记不删除"),
    ("high_spread_median", "高传播(中位数)标记", "log_view_per_day>median", "int(0/1)", "标记变量", "—",
     "高于日均播放中位数"),
    ("high_spread_q75", "高传播(前25%)标记", "log_view_per_day>=P75", "int(0/1)", "标记变量", "—",
     "日均播放前 25%"),
]
dict_df = pd.DataFrame(DICT_ROWS, columns=[
    "variable_en", "alias_cn", "formula", "dtype", "role_cn", "dv_type", "note_cn"])
meta_df = pd.DataFrame({
    "key": ["生成时间", "输入文件", "样本量(行)", "标准化口径(ddof)", "极端值处理", "脚本"],
    "value": [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), INPUT_PATH, N0,
              f"ddof={ZSCORE_DDOF}（总体标准差）", "仅标记不删除", os.path.basename(__file__)
              if "__file__" in globals() else "step2_propagation_metrics.py"],
})
with pd.ExcelWriter(OUT_DICT_XLSX, engine="openpyxl") as xw:
    dict_df.to_excel(xw, sheet_name="variable_dictionary", index=False)
    meta_df.to_excel(xw, sheet_name="meta", index=False)
qc("output::dictionary_xlsx", "OK", OUT_DICT_XLSX)


# ------------------------------------------------------------------ #
# 10. 保存主数据（不覆盖输入）
# ------------------------------------------------------------------ #
if os.path.abspath(OUT_METRICS_CSV) == os.path.abspath(INPUT_PATH):
    qc("output::metrics_csv", "FAIL", "输出路径等于输入路径，已阻止覆盖")
    logger.error("拒绝覆盖输入文件，终止保存。")
    sys.exit(1)
df.to_csv(OUT_METRICS_CSV, index=False, encoding="utf-8-sig")
qc("output::metrics_csv", "OK", f"{OUT_METRICS_CSV}（{df.shape[0]} 行 × {df.shape[1]} 列）")

# 行数守恒校验
if len(df) != N0:
    qc("row_conservation", "FAIL", f"行数从 {N0} 变为 {len(df)}（本步骤不应增删行）")
else:
    qc("row_conservation", "OK", f"{N0} 行保持不变")


# ------------------------------------------------------------------ #
# 11. 收尾
# ------------------------------------------------------------------ #
n_fail = sum(r["status"] == "FAIL" for r in QC_ROWS)
n_warn = sum(r["status"] == "WARN" for r in QC_ROWS)
logger.info("-" * 70)
logger.info(f"质量检查汇总：FAIL={n_fail}, WARN={n_warn}, OK={len(QC_ROWS)-n_fail-n_warn}")
logger.info("产物清单：")
for p in (OUT_METRICS_CSV, OUT_SUMMARY_XLSX, OUT_DICT_XLSX,
          FIG_LOGVIEW, FIG_LOGVPD, FIG_RECOG, FIG_PARTI, LOG_PATH):
    logger.info(f"  - {p}")
logger.info("步骤二完成。")
print("\nDONE. FAIL=%d WARN=%d" % (n_fail, n_warn))
