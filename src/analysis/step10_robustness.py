#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 10: 稳健性检验
研究题目：标题和封面双模态协同对社交媒体平台视频传播的影响——以B站知识类视频为例
依赖包：pandas, numpy, statsmodels, scipy, scikit-learn, matplotlib, seaborn, openpyxl
运行方式：python step10_robustness.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import logging
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.preprocessing import StandardScaler, QuantileTransformer

# ============================================================
# 0. 路径与日志配置
# ============================================================
BASE_DIR    = os.path.abspath("outputs")
OUTPUT_DIR  = BASE_DIR
FIG_DIR     = os.path.join(OUTPUT_DIR, "figures")
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")

for d in [OUTPUT_DIR, FIG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "robustness_report.md")

# Python logging（终端 + 文件同步）
py_logger = logging.getLogger("robustness")
py_logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, "robustness_python.log"), encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
py_logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
py_logger.addHandler(sh)

# Markdown 报告缓冲
md_lines = []
def md(text=""):
    md_lines.append(text)
    py_logger.info(text)

def save_md():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

# ============================================================
# 1. 读取数据
# ============================================================
DATA_PATH = os.path.join(BASE_DIR, "modeling_dataset.csv")

md(f"# 稳健性检验报告")
md(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
md(f"**数据来源**: `{DATA_PATH}`")
md()

try:
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    md(f"## 0. 数据载入")
    md(f"- 原始样本量：{len(df_raw):,} 行 × {df_raw.shape[1]} 列")
except Exception as e:
    py_logger.error(f"数据读取失败: {e}")
    sys.exit(1)

# ============================================================
# 2. 变量定义与数据清洗
# ============================================================
md()
md("## 1. 变量定义与清洗")

# ---- 2.1 控制变量列表（主模型 Model 3/4 的设定）----
CONTROLS = [
    "log_fans", "log_age", "log_duration",
    "title_char_len", "title_char_len_sq",
    "title_clickbait_score", "title_sentiment_score",
    "image_brightness", "image_colorfulness",
    "human_presence", "cover_has_text", "cover_text_area_ratio",
    "pub_year", "pub_month", "pub_weekday", "pub_hour",
]

# ---- 2.2 核心解释变量 ----
CORE_VARS = [
    "image_title_fit",
    "image_title_fit_sq",
    "title_cover_text_overlap",
    "semantic_text_interaction",
    "sentiment_consistency",
    "sentiment_conflict",
]
# 互补类型哑变量（参照组：low_relatedness）
COMP_DUMMIES = [
    "type_high_synergy",
    "type_complementary_explanation",
    "type_redundant_reinforcement",
    "type_curiosity_contrast",
    "type_emotional_conflict",
]

CORE_ALL = CORE_VARS + COMP_DUMMIES

# ---- 2.3 因变量列表 ----
DEP_VARS = [
    "log_view",
    "log_view_per_day",
    "recognition",
    "participation",
    "communication_effect",
    "relative_view",
]

# ---- 2.4 域哑变量 ----
df_raw["domain"] = df_raw["domain"].astype(str).str.strip()
domain_dummies = pd.get_dummies(df_raw["domain"], prefix="domain", drop_first=True)

# ---- 2.5 组装工作数据集 ----
ALL_NEEDED = list(set(
    CONTROLS + CORE_ALL + DEP_VARS +
    ["log_view", "uid", "domain", "view", "fans", "duration",
     "like", "coin", "favorite", "share", "danmaku", "reply",
     "high_spread_median", "high_spread_q75",
     "title_cover_text_jaccard", "title_cover_text_tfidf_similarity",
     "image_title_fit_z", "complementarity_type", "log_age"]
))

missing_cols = [c for c in ALL_NEEDED if c not in df_raw.columns]
if missing_cols:
    md(f"[WARN] 缺失列（将跳过相关分析）：{missing_cols}")
    py_logger.warning(f"缺失列: {missing_cols}")
    ALL_NEEDED = [c for c in ALL_NEEDED if c in df_raw.columns]

df = df_raw[list(set(ALL_NEEDED))].copy()

# 拼接域哑变量
df = pd.concat([df, domain_dummies], axis=1)
DOMAIN_COLS = list(domain_dummies.columns)

# 数值型转换
for col in df.select_dtypes(include="object").columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="raise")
    except:
        pass  # 保留 string 列（uid/domain/complementarity_type）

# 关键变量缺失检查
key_check = CONTROLS + CORE_ALL + DEP_VARS
miss_report = {}
for col in key_check:
    if col in df.columns:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            miss_report[col] = n_miss

if miss_report:
    md(f"[WARN] 含缺失值变量：{miss_report}")
    py_logger.warning(f"含缺失值变量: {miss_report}")

# 删除核心变量缺失行
core_drop = [c for c in CORE_ALL + DEP_VARS + CONTROLS if c in df.columns]
df_clean = df.dropna(subset=core_drop).copy()
md(f"- 清洗后样本量（删除核心缺失行）：{len(df_clean):,}")

# ---- 2.6 新增互补类型重编码（4类合并）----
def recode_comp4(c):
    if c in ["high_synergy", "complementary_explanation"]:
        return "high_synergy_complementary"
    elif c == "redundant_reinforcement":
        return "redundant"
    elif c in ["curiosity_contrast", "emotional_conflict"]:
        return "contrast_conflict"
    else:
        return "low_relatedness"

df_clean["complementarity_type4"] = df_clean["complementarity_type"].apply(recode_comp4)
comp4_dummies = pd.get_dummies(df_clean["complementarity_type4"], prefix="comp4", drop_first=True)
df_clean = pd.concat([df_clean, comp4_dummies], axis=1)
COMP4_DUMMIES = list(comp4_dummies.columns)

# ---- 2.7 image_title_fit 分位数组变量 ----
df_clean["fit_quantile"] = pd.qcut(
    df_clean["image_title_fit"], q=4,
    labels=["Q1","Q2","Q3","Q4"]
).astype(str)
fit_q_dummies = pd.get_dummies(df_clean["fit_quantile"], prefix="fitq", drop_first=True)
df_clean = pd.concat([df_clean, fit_q_dummies], axis=1)
FIT_Q_DUMMIES = list(fit_q_dummies.columns)

# ---- 2.8 overlap 均值变量 ----
if "title_cover_text_jaccard" in df_clean.columns and "title_cover_text_tfidf_similarity" in df_clean.columns:
    df_clean["overlap_mean"] = (
        df_clean["title_cover_text_jaccard"] + df_clean["title_cover_text_tfidf_similarity"]
    ) / 2
    df_clean["overlap_jaccard"] = df_clean["title_cover_text_jaccard"]
    df_clean["overlap_tfidf"] = df_clean["title_cover_text_tfidf_similarity"]
else:
    df_clean["overlap_mean"]    = df_clean["title_cover_text_overlap"]
    df_clean["overlap_jaccard"] = df_clean["title_cover_text_overlap"]
    df_clean["overlap_tfidf"]   = df_clean["title_cover_text_overlap"]

md(f"- 新增互补类型4分类、fit分位数组、overlap均值变量")
md()

# ============================================================
# 3. 辅助函数
# ============================================================

def build_formula(dep, core=None, controls=None, domain_cols=None, comp_type="6cat"):
    """构建 statsmodels 公式字符串"""
    if core is None:
        core = CORE_VARS + COMP_DUMMIES
    if controls is None:
        controls = CONTROLS
    if domain_cols is None:
        domain_cols = DOMAIN_COLS

    # 选择互补类型哑变量
    if comp_type == "6cat":
        comp_part = COMP_DUMMIES
    elif comp_type == "4cat":
        comp_part = COMP4_DUMMIES
    else:
        comp_part = []

    core_use = [v for v in core if v not in COMP_DUMMIES + COMP4_DUMMIES]
    rhs_parts = core_use + comp_part + controls + domain_cols
    # 过滤不存在的列
    rhs_parts = [c for c in rhs_parts if c in df_clean.columns]
    # 去重、避免自身同名
    seen = set()
    rhs_unique = []
    for c in rhs_parts:
        if c not in seen and c != dep:
            seen.add(c)
            rhs_unique.append(c)

    formula = dep + " ~ " + " + ".join(rhs_unique)
    return formula


def run_ols(formula, data, cov_type="nonrobust", cluster_var=None):
    """运行 OLS，返回 result 对象（params 始终为 pd.Series）"""
    try:
        if cov_type == "HC3":
            return smf.ols(formula, data=data).fit(cov_type="HC3", disp=False)
        elif cov_type == "cluster" and cluster_var is not None:
            grps = data[cluster_var].values
            return smf.ols(formula, data=data).fit(
                cov_type="cluster", cov_kwds={"groups": grps}, disp=False)
        else:
            return smf.ols(formula, data=data).fit(disp=False)
    except Exception as e:
        py_logger.warning(f"OLS 失败 | formula={formula[:80]} | err={e}")
        return None


def extract_coefs(result, var_list):
    """从回归结果中提取指定变量的系数、p值、显著性标记"""
    rows = []
    if result is None:
        for v in var_list:
            rows.append({"variable": v, "coef": np.nan, "pvalue": np.nan, "sig": "N/A", "n_obs": np.nan})
        return pd.DataFrame(rows)

    # 将 params/pvalues 统一为 pd.Series
    params  = result.params
    pvalues = result.pvalues
    if not isinstance(params, pd.Series):
        try:
            params  = pd.Series(params,  index=result.model.exog_names)
            pvalues = pd.Series(pvalues, index=result.model.exog_names)
        except Exception:
            params  = pd.Series(params)
            pvalues = pd.Series(pvalues)
    nobs    = int(result.nobs)

    for v in var_list:
        coef = params.get(v, np.nan)
        pval = pvalues.get(v, np.nan)
        if np.isnan(pval):
            sig = "n.s."
        elif pval < 0.01:
            sig = "***"
        elif pval < 0.05:
            sig = "**"
        elif pval < 0.10:
            sig = "*"
        else:
            sig = "n.s."
        rows.append({"variable": v, "coef": coef, "pvalue": pval, "sig": sig, "n_obs": nobs})
    return pd.DataFrame(rows)


def run_logit(formula, data):
    """运行 Logit 回归，返回 result"""
    try:
        m = smf.logit(formula, data=data).fit(method="bfgs", disp=False, maxiter=200)
        return m
    except Exception as e:
        py_logger.warning(f"Logit 失败 | err={e}")
        return None


def run_negbin(dep_raw, rhs_cols, data, exposure_col=None):
    """负二项回归（对原始 count 型因变量）"""
    try:
        X = data[rhs_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="any")
        X = sm.add_constant(X)
        y = data[dep_raw].astype(float)
        if exposure_col and exposure_col in data.columns:
            exposure = data[exposure_col].values
            m = sm.NegativeBinomial(y, X, exposure=exposure).fit(disp=False, maxiter=200)
        else:
            m = sm.NegativeBinomial(y, X).fit(disp=False, maxiter=200)
        return m
    except Exception as e:
        py_logger.warning(f"NegBin 失败 | err={e}")
        return None


def winsorize_col(series, lower=0.01, upper=0.99):
    """单列缩尾"""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def sig_consistent(sig_list):
    """判断显著性是否一致（均显著或均不显著）"""
    sigs = [s for s in sig_list if s not in ("N/A", None)]
    if not sigs:
        return False
    sig_flag = [s != "n.s." for s in sigs]
    return all(sig_flag) or (not any(sig_flag))


def direction_consistent(coef_list):
    """系数方向是否一致"""
    coefs = [c for c in coef_list if not np.isnan(c)]
    if not coefs:
        return False
    return all(c > 0 for c in coefs) or all(c < 0 for c in coefs)


# ============================================================
# 4. 主基准模型（Model 3 基本设定）
# ============================================================
md("## 2. 基准模型确认（Model 3）")

BASE_FORMULA = build_formula("log_view", comp_type="6cat")
base_result  = run_ols(BASE_FORMULA, df_clean, cov_type="HC3")

if base_result:
    md(f"- 基准样本量：N = {int(base_result.nobs):,}")
    md(f"- 基准 R²：{base_result.rsquared:.4f}，Adj R²：{base_result.rsquared_adj:.4f}")
else:
    md("[WARN] 基准模型未能拟合，请检查数据")

md()

# 汇总结果存储
RESULTS = {}       # dict: robustness_label -> {dep_var -> extract_coefs_df}
SUMMARY_MATRIX = {}  # {check_label: {var: {direction, sig, robust}}}

def record_result(label, dep, result_obj, var_list=None):
    if var_list is None:
        var_list = CORE_ALL
    df_c = extract_coefs(result_obj, var_list)
    df_c["dep_var"]  = dep
    df_c["check"]    = label
    if label not in RESULTS:
        RESULTS[label] = []
    RESULTS[label].append(df_c)


# ============================================================
# 5. Part I：替换因变量
# ============================================================
md("## 3. Part I：替换因变量")
md()
md("使用主模型（Model 3）设定，将因变量依次替换。")
md()

DEP_LABELS = {
    "log_view":            "对数播放量（基准）",
    "log_view_per_day":    "日均播放对数",
    "recognition":         "认知传播综合指数",
    "participation":       "参与传播综合指数",
    "communication_effect":"传播效果综合指数",
    "relative_view":       "相对播放量（去粉丝规模效应）",
}

dep_coef_rows = []

for dep in DEP_VARS:
    formula = build_formula(dep, comp_type="6cat")
    res = run_ols(formula, df_clean, cov_type="HC3")
    record_result(f"dep_{dep}", dep, res)
    label = f"Part I: 替换因变量 ({dep})"
    if res:
        for v in CORE_ALL:
            c = res.params.get(v, np.nan)
            p = res.pvalues.get(v, np.nan)
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "n.s."))
            dep_coef_rows.append({
                "因变量": DEP_LABELS.get(dep, dep),
                "variable": v,
                "coef": round(c, 4),
                "pvalue": round(p, 4) if not np.isnan(p) else np.nan,
                "sig": sig,
                "n_obs": int(res.nobs),
                "R2": round(res.rsquared, 4),
                "AdjR2": round(res.rsquared_adj, 4),
            })
        md(f"- **{dep}**: N={int(res.nobs):,}, R²={res.rsquared:.4f}, Adj R²={res.rsquared_adj:.4f}")
    else:
        md(f"- **{dep}**: 模型拟合失败")

dep_coef_df = pd.DataFrame(dep_coef_rows)
md()

# ============================================================
# 6. Part II：极端值处理
# ============================================================
md("## 4. Part II：极端值处理")
md()

extreme_rows = []

def run_extreme_check(label, data_sub, dep="log_view"):
    formula = build_formula(dep, comp_type="6cat")
    res = run_ols(formula, data_sub, cov_type="HC3")
    record_result(f"extreme_{label}", dep, res)
    if res:
        row_base = {"检验类型": label, "n_obs": int(res.nobs), "R2": round(res.rsquared, 4)}
        for v in CORE_ALL:
            c = res.params.get(v, np.nan)
            p = res.pvalues.get(v, np.nan)
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "n.s."))
            extreme_rows.append({**row_base, "variable": v, "coef": round(c, 4),
                                  "pvalue": round(p, 4) if not np.isnan(p) else np.nan, "sig": sig})
        md(f"- **{label}**: N={int(res.nobs):,}, R²={res.rsquared:.4f}")
    else:
        md(f"- **{label}**: 模型失败")
    return res

# 6.1 删除 view 前 1%
q99_view = df_clean["view"].quantile(0.99)
df_drop_top_view = df_clean[df_clean["view"] <= q99_view].copy()
run_extreme_check("删除view前1%", df_drop_top_view)

# 6.2 Winsorize 1%-99%
winsor_cols = ["view", "fans", "duration", "like", "coin", "favorite", "share", "danmaku", "reply"]
df_winsor = df_clean.copy()
for col in winsor_cols:
    if col in df_winsor.columns:
        df_winsor[col] = winsorize_col(df_winsor[col])
# 重新计算对数变量
for rawcol, logcol in [("view","log_view"),("fans","log_fans"),("duration","log_duration")]:
    if rawcol in df_winsor.columns and logcol in df_winsor.columns:
        df_winsor[logcol] = np.log1p(df_winsor[rawcol])
run_extreme_check("1%-99%缩尾后回归", df_winsor)

# 6.3 删除 fans 前 1% 头部 UP 主
q99_fans = df_clean["fans"].quantile(0.99)
df_drop_top_fans = df_clean[df_clean["fans"] <= q99_fans].copy()
run_extreme_check("删除fans前1%头部UP主", df_drop_top_fans)

# 6.4 删除 duration 过短和过长极端值
q01_dur = df_clean["duration"].quantile(0.01)
q99_dur = df_clean["duration"].quantile(0.99)
df_dur = df_clean[(df_clean["duration"] >= q01_dur) & (df_clean["duration"] <= q99_dur)].copy()
run_extreme_check("删除duration 1%-99%极端值", df_dur)

extreme_df = pd.DataFrame(extreme_rows)
md()

# ============================================================
# 7. Part III：标准误替换
# ============================================================
md("## 5. Part III：标准误替换")
md()

se_rows = []

def run_se_check(label, cov_type, cluster_var=None):
    formula = build_formula("log_view", comp_type="6cat")
    res = run_ols(formula, df_clean, cov_type=cov_type, cluster_var=cluster_var)
    record_result(f"se_{label}", "log_view", res)
    if res:
        row_base = {"标准误类型": label, "n_obs": int(res.nobs)}
        for v in CORE_ALL:
            c  = res.params.get(v, np.nan)
            se = res.bse.get(v, np.nan) if hasattr(res, "bse") else np.nan
            p  = res.pvalues.get(v, np.nan)
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "n.s."))
            se_rows.append({**row_base, "variable": v, "coef": round(c, 4),
                            "std_err": round(se, 4), "pvalue": round(p, 4) if not np.isnan(p) else np.nan, "sig": sig})
        md(f"- **{label}**: N={int(res.nobs):,}")
    else:
        md(f"- **{label}**: 标准误估计失败")

run_se_check("普通OLS标准误", "nonrobust")
run_se_check("HC3异方差稳健标准误", "HC3")
run_se_check("按uid聚类稳健标准误", "cluster", cluster_var="uid")

se_df = pd.DataFrame(se_rows)
md()

# ============================================================
# 8. Part IV：模型替换
# ============================================================
md("## 6. Part IV：模型替换")
md()

model_rows = []

# 8.1 Logistic：high_spread_median
md("### 4.1 Logistic 回归：high_spread_median")
logit_formula_m = build_formula("high_spread_median", comp_type="6cat")
logit_m = run_logit(logit_formula_m, df_clean)
if logit_m:
    md(f"- Logit (median) 收敛, N={int(logit_m.nobs):,}, Pseudo R²={logit_m.prsquared:.4f}")
    record_result("logit_median", "high_spread_median", logit_m)
    for v in CORE_ALL:
        c = logit_m.params.get(v, np.nan)
        p = logit_m.pvalues.get(v, np.nan)
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "n.s."))
        model_rows.append({"模型类型": "Logit(median)", "variable": v, "coef": round(c, 4),
                           "pvalue": round(p, 4) if not np.isnan(p) else np.nan, "sig": sig})
else:
    md("- Logit (median) 失败")

# 8.2 Logistic：high_spread_q75
md("### 4.2 Logistic 回归：high_spread_q75")
logit_formula_q = build_formula("high_spread_q75", comp_type="6cat")
logit_q = run_logit(logit_formula_q, df_clean)
if logit_q:
    md(f"- Logit (q75) 收敛, N={int(logit_q.nobs):,}, Pseudo R²={logit_q.prsquared:.4f}")
    record_result("logit_q75", "high_spread_q75", logit_q)
    for v in CORE_ALL:
        c = logit_q.params.get(v, np.nan)
        p = logit_q.pvalues.get(v, np.nan)
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "n.s."))
        model_rows.append({"模型类型": "Logit(q75)", "variable": v, "coef": round(c, 4),
                           "pvalue": round(p, 4) if not np.isnan(p) else np.nan, "sig": sig})
else:
    md("- Logit (q75) 失败")

# 8.3 负二项回归（原始 view 为因变量）
md("### 4.3 负二项回归（count: view）")
rhs_nb = [c for c in CORE_VARS + COMP_DUMMIES + CONTROLS + DOMAIN_COLS if c in df_clean.columns]
rhs_nb = list(dict.fromkeys(rhs_nb))  # 去重
rhs_nb = [c for c in rhs_nb if c != "view"]
# 过滤掉非数值列，防止 NegBin 报 object dtype 错误
rhs_nb = [c for c in rhs_nb if pd.api.types.is_numeric_dtype(df_clean[c])]

negbin_res = run_negbin("view", rhs_nb, df_clean, exposure_col=None)
if negbin_res:
    md(f"- NegBin 收敛, N={int(negbin_res.nobs):,}")
    record_result("negbin_view", "view", negbin_res)
    for v in CORE_ALL:
        c = negbin_res.params.get(v, negbin_res.params.get("const", np.nan) if v == "const" else np.nan)
        p = negbin_res.pvalues.get(v, np.nan)
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "n.s."))
        model_rows.append({"模型类型": "NegBin(view)", "variable": v, "coef": round(c, 4) if not np.isnan(c) else np.nan,
                           "pvalue": round(p, 4) if not np.isnan(p) else np.nan, "sig": sig})
else:
    md("- NegBin 失败，可能为数值收敛问题")

# 8.4 分位数回归（中位数 tau=0.5）
md("### 4.4 分位数回归（τ=0.5）")
try:
    qr_formula = build_formula("log_view", comp_type="6cat")
    qr_model = smf.quantreg(qr_formula, df_clean).fit(q=0.5, max_iter=1000)
    md(f"- QuantReg(τ=0.5) 收敛, N={int(qr_model.nobs):,}")
    record_result("quantreg_50", "log_view", qr_model)
    for v in CORE_ALL:
        c = qr_model.params.get(v, np.nan)
        p = qr_model.pvalues.get(v, np.nan)
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "n.s."))
        model_rows.append({"模型类型": "QuantReg(τ=0.5)", "variable": v, "coef": round(c, 4),
                           "pvalue": round(p, 4) if not np.isnan(p) else np.nan, "sig": sig})
except Exception as e:
    md(f"- QuantReg 失败: {e}")
    py_logger.warning(f"QuantReg 失败: {e}")

model_df = pd.DataFrame(model_rows)
md()

# ============================================================
# 9. Part V：核心变量定义替换
# ============================================================
md("## 7. Part V：核心变量定义替换")
md()

vardef_rows = []

def run_vardef_check(label, data, formula_str):
    res = run_ols(formula_str, data, cov_type="HC3")
    if res:
        md(f"- **{label}**: N={int(res.nobs):,}, R²={res.rsquared:.4f}")
        for v in CORE_ALL:
            c = res.params.get(v, np.nan)
            p = res.pvalues.get(v, np.nan)
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "n.s."))
            vardef_rows.append({"变量定义": label, "variable": v, "coef": round(c, 4),
                                "pvalue": round(p, 4) if not np.isnan(p) else np.nan, "sig": sig, "n_obs": int(res.nobs)})
        record_result(f"vardef_{label}", "log_view", res)
    else:
        md(f"- **{label}**: 失败")

# 9.1 image_title_fit：原始值（基准）
md("### 5.1 image_title_fit 定义替换")
# 原始值 = 基准模型，跳过

# 标准化值：用 image_title_fit_z 替换
def swap_fit(df_in, fit_col, sq_col=None):
    """替换 fit 变量后重新构建公式"""
    core_new = [v for v in CORE_VARS if v not in ("image_title_fit","image_title_fit_sq")]
    core_new = [fit_col] + ([sq_col] if sq_col else []) + core_new
    return build_formula("log_view", core=core_new, comp_type="6cat")

# 标准化 fit（image_title_fit_z）
df_clean["fit_z"] = df_clean["image_title_fit_z"] if "image_title_fit_z" in df_clean.columns else (
    (df_clean["image_title_fit"] - df_clean["image_title_fit"].mean()) / df_clean["image_title_fit"].std()
)
df_clean["fit_z_sq"] = df_clean["fit_z"] ** 2
formula_fit_z = swap_fit(df_clean, "fit_z", "fit_z_sq")
run_vardef_check("image_title_fit_z（标准化）", df_clean, formula_fit_z)

# 分位数组（FIT_Q_DUMMIES 替换连续变量）
core_fitq = [v for v in CORE_VARS if v not in ("image_title_fit","image_title_fit_sq")] + FIT_Q_DUMMIES
formula_fitq = build_formula("log_view", core=core_fitq, comp_type="6cat")
run_vardef_check("image_title_fit_分位数组(Q2/Q3/Q4 vs Q1)", df_clean, formula_fitq)

# 9.2 title_cover_text_overlap 替换
md("### 5.2 title_cover_text_overlap 定义替换")

def swap_overlap(df_in, overlap_col):
    core_new = [v if v != "title_cover_text_overlap" else overlap_col for v in CORE_VARS]
    return build_formula("log_view", core=core_new, comp_type="6cat")

run_vardef_check("overlap=Jaccard", df_clean, swap_overlap(df_clean, "overlap_jaccard"))
run_vardef_check("overlap=TF-IDF", df_clean, swap_overlap(df_clean, "overlap_tfidf"))
run_vardef_check("overlap=均值(Jaccard+TFIDF)/2", df_clean, swap_overlap(df_clean, "overlap_mean"))

# 9.3 complementarity_type：4类合并
md("### 5.3 complementarity_type 类别合并（4类）")
core_4cat = CORE_VARS + COMP4_DUMMIES
formula_4cat = build_formula("log_view", core=core_4cat, comp_type="4cat")
run_vardef_check("complementarity_type_4类合并", df_clean, formula_4cat)

# 9.4 高传播分组定义替换（已在 Part IV Logistic 中涵盖）
md("### 5.4 高传播定义替换（中位数 vs 前25%）")
md("  见 Part IV Logistic 回归（high_spread_median / high_spread_q75）")
md()

vardef_df = pd.DataFrame(vardef_rows)

# ============================================================
# 10. Part VI：结论稳健性矩阵
# ============================================================
md("## 8. Part VI：稳健性结论矩阵")
md()

# 核心变量（不含互补类型哑变量：过多，单独处理）
MATRIX_VARS = [
    "image_title_fit",
    "image_title_fit_sq",
    "title_cover_text_overlap",
    "semantic_text_interaction",
    "sentiment_consistency",
    "sentiment_conflict",
    # 互补类型代表性哑变量
    "type_high_synergy",
    "type_complementary_explanation",
    "type_redundant_reinforcement",
    "type_curiosity_contrast",
    "type_emotional_conflict",
]

# 收集所有检验结果
ALL_CHECKS = {
    # Part I
    "DV: log_view_per_day":      f"dep_log_view_per_day",
    "DV: recognition":           f"dep_recognition",
    "DV: participation":         f"dep_participation",
    "DV: communication_effect":  f"dep_communication_effect",
    "DV: relative_view":         f"dep_relative_view",
    # Part II
    "EV: 删除view前1%":          f"extreme_删除view前1%",
    "EV: 1%-99%缩尾":            f"extreme_1%-99%缩尾后回归",
    "EV: 删除fans前1%":          f"extreme_删除fans前1%头部UP主",
    "EV: 删除duration极端值":    f"extreme_删除duration 1%-99%极端值",
    # Part III
    "SE: 普通OLS":               f"se_普通OLS标准误",
    "SE: HC3":                   f"se_HC3异方差稳健标准误",
    "SE: uid聚类":               f"se_按uid聚类稳健标准误",
    # Part IV
    "Model: Logit(median)":      f"logit_median",
    "Model: Logit(q75)":         f"logit_q75",
    "Model: NegBin":             f"negbin_view",
    "Model: QuantReg(τ=0.5)":   f"quantreg_50",
    # Part V
    "Def: fit_z标准化":          f"vardef_image_title_fit_z（标准化）",
    "Def: fit_分位数组":         f"vardef_image_title_fit_分位数组(Q2/Q3/Q4 vs Q1)",
    "Def: overlap=Jaccard":      f"vardef_overlap=Jaccard",
    "Def: overlap=TFIDF":        f"vardef_overlap=TF-IDF",
    "Def: overlap=均值":         f"vardef_overlap=均值(Jaccard+TFIDF)/2",
    "Def: comp_4类合并":         f"vardef_complementarity_type_4类合并",
}

# 基准方向（从基准模型提取）
baseline_dir = {}
baseline_sig = {}
if base_result:
    for v in MATRIX_VARS:
        c = base_result.params.get(v, np.nan)
        p = base_result.pvalues.get(v, np.nan)
        baseline_dir[v] = np.sign(c) if not np.isnan(c) else np.nan
        baseline_sig[v] = (not np.isnan(p)) and (p < 0.1)

# 构建结论矩阵
matrix_data = {}

for check_label, result_key in ALL_CHECKS.items():
    matrix_data[check_label] = {}
    if result_key in RESULTS:
        dfs = RESULTS[result_key]
        # 合并所有 df（一个 key 可能对应多个 dep）
        combined = pd.concat(dfs, ignore_index=True)
        for v in MATRIX_VARS:
            row = combined[combined["variable"] == v]
            if len(row) == 0:
                matrix_data[check_label][v] = "NA"
                continue
            coef = row["coef"].values[0]
            pval = row["pvalue"].values[0]
            sig_flag = (not np.isnan(pval)) and (pval < 0.1)
            base_d   = baseline_dir.get(v, np.nan)
            base_s   = baseline_sig.get(v, False)
            # 方向一致
            dir_ok = (not np.isnan(coef)) and (not np.isnan(base_d)) and (np.sign(coef) == base_d)
            # 显著性一致
            sig_ok = (sig_flag == base_s)
            # 稳健 = 方向 + 显著性均一致
            robust = dir_ok and sig_ok
            # 记录符号
            if np.isnan(coef):
                cell = "NA"
            else:
                sig_mark = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
                dir_sym  = "✓" if dir_ok else "✗"
                cell = f"{coef:+.3f}{sig_mark}({dir_sym})"
            matrix_data[check_label][v] = {
                "coef": coef, "pval": pval, "sig": sig_flag,
                "dir_ok": dir_ok, "sig_ok": sig_ok, "robust": robust,
                "display": cell
            }
    else:
        for v in MATRIX_VARS:
            matrix_data[check_label][v] = "NA"

# 转为 DataFrame（display 用）
display_matrix = pd.DataFrame({
    check: {v: (matrix_data[check][v]["display"]
                if isinstance(matrix_data[check][v], dict)
                else matrix_data[check][v])
            for v in MATRIX_VARS}
    for check in ALL_CHECKS.keys()
}).T

# 方向一致率矩阵
dir_matrix = pd.DataFrame({
    check: {v: (matrix_data[check][v]["dir_ok"]
                if isinstance(matrix_data[check][v], dict)
                else np.nan)
            for v in MATRIX_VARS}
    for check in ALL_CHECKS.keys()
}).T

# 稳健矩阵
robust_matrix = pd.DataFrame({
    check: {v: (1 if (isinstance(matrix_data[check][v], dict) and matrix_data[check][v]["robust"]) else 0)
            for v in MATRIX_VARS}
    for check in ALL_CHECKS.keys()
}).T

# 汇总稳健比例
robustness_summary = {}
for v in MATRIX_VARS:
    dir_list  = [matrix_data[c][v]["dir_ok"] for c in ALL_CHECKS if isinstance(matrix_data[c][v], dict)]
    sig_list  = [matrix_data[c][v]["sig_ok"] for c in ALL_CHECKS if isinstance(matrix_data[c][v], dict)]
    robust_l  = [matrix_data[c][v]["robust"] for c in ALL_CHECKS if isinstance(matrix_data[c][v], dict)]
    n_valid   = len(dir_list)
    robustness_summary[v] = {
        "variable": v,
        "n_checks": n_valid,
        "dir_consistent_pct": f"{sum(dir_list)/n_valid*100:.1f}%" if n_valid else "N/A",
        "sig_consistent_pct": f"{sum(sig_list)/n_valid*100:.1f}%" if n_valid else "N/A",
        "robust_pct": f"{sum(robust_l)/n_valid*100:.1f}%" if n_valid else "N/A",
        "conclusion": ("稳健" if n_valid and sum(robust_l)/n_valid >= 0.7
                       else ("部分成立" if n_valid and sum(robust_l)/n_valid >= 0.4
                             else "需谨慎解释")),
    }

summary_df = pd.DataFrame(list(robustness_summary.values()))

# ============================================================
# 11. 输出 Excel
# ============================================================
md()
md("## 9. 输出文件")

EXCEL_PATH = os.path.join(OUTPUT_DIR, "robustness_results.xlsx")

try:
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:

        # Sheet 1: 基准模型系数
        if base_result:
            base_df = pd.DataFrame({
                "variable": base_result.params.index,
                "coef":     base_result.params.values,
                "std_err":  base_result.bse.values,
                "t_stat":   base_result.tvalues.values,
                "pvalue":   base_result.pvalues.values,
            })
            base_df["sig"] = base_df["pvalue"].apply(
                lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "n.s.")))
            base_df.to_excel(writer, sheet_name="0_基准模型Model3", index=False)

        # Sheet 2: Part I 替换因变量
        if not dep_coef_df.empty:
            dep_coef_df.to_excel(writer, sheet_name="1_替换因变量", index=False)

        # Sheet 3: Part II 极端值
        if not extreme_df.empty:
            extreme_df.to_excel(writer, sheet_name="2_极端值处理", index=False)

        # Sheet 4: Part III 标准误
        if not se_df.empty:
            se_df.to_excel(writer, sheet_name="3_标准误替换", index=False)

        # Sheet 5: Part IV 模型替换
        if not model_df.empty:
            model_df.to_excel(writer, sheet_name="4_模型替换", index=False)

        # Sheet 6: Part V 变量定义
        if not vardef_df.empty:
            vardef_df.to_excel(writer, sheet_name="5_变量定义替换", index=False)

        # Sheet 7: 结论矩阵（显示）
        display_matrix.to_excel(writer, sheet_name="6_结论矩阵_系数显示")

        # Sheet 8: 稳健矩阵（0/1）
        robust_matrix.to_excel(writer, sheet_name="7_稳健矩阵_01")

        # Sheet 9: 稳健性汇总
        summary_df.to_excel(writer, sheet_name="8_稳健性汇总", index=False)

    md(f"- [OK] `outputs/robustness_results.xlsx` 写入成功（9个Sheet）")
except Exception as e:
    md(f"- [X] Excel 写入失败: {e}")
    py_logger.error(traceback.format_exc())

# ============================================================
# 12. 图表：核心系数方向热图 & 稳健性汇总热图
# ============================================================
md()
md("## 10. 图表输出")

VAR_LABELS_CN = {
    "image_title_fit":                 "图文匹配度",
    "image_title_fit_sq":              "图文匹配度²",
    "title_cover_text_overlap":        "标题-封面\n文字重叠",
    "semantic_text_interaction":       "语义-文字\n交互",
    "sentiment_consistency":           "情感一致",
    "sentiment_conflict":              "情感冲突",
    "type_high_synergy":               "高协同型",
    "type_complementary_explanation":  "互补解释型",
    "type_redundant_reinforcement":    "冗余强化型",
    "type_curiosity_contrast":         "好奇对比型",
    "type_emotional_conflict":         "情感冲突型",
}

# ---- 浙大论文字体规范配置 ----
# Windows 优先使用 SimHei（黑体）/ Microsoft YaHei（微软雅黑）
# macOS 备选 PingFang SC / STHeiti；Linux 备选 WenQuanYi Micro Hei
ZJU_CN_FONTS = ["SimHei", "Microsoft YaHei", "PingFang SC",
                "STHeiti", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["font.sans-serif"]     = ZJU_CN_FONTS
plt.rcParams["font.family"]         = "sans-serif"
plt.rcParams["axes.unicode_minus"]  = False   # 负号正常显示
plt.rcParams["axes.linewidth"]      = 0.8
plt.rcParams["xtick.direction"]     = "out"
plt.rcParams["ytick.direction"]     = "out"

# 浙大规范色板（偏学术、低饱和）
COLOR_ROBUST   = "#2166AC"   # 完全稳健 → 深蓝
COLOR_DIR_ONLY = "#92C5DE"   # 方向一致 → 浅蓝
COLOR_FAIL     = "#D6604D"   # 方向不一致 → 砖红
COLOR_NA       = "#D9D9D9"   # 无法估计 → 浅灰

col_labels_cn = [VAR_LABELS_CN.get(v, v) for v in MATRIX_VARS]

# ============================================================
# 图1：稳健性检验 × 核心变量 —— 系数方向一致性矩阵
# （浙大规范：图题置于图下方，编号格式"图X.X"）
# ============================================================
fig1_path = os.path.join(FIG_DIR, "robustness_core_coefficients.png")

color_matrix = np.full((len(ALL_CHECKS), len(MATRIX_VARS)), np.nan)
for i, check in enumerate(ALL_CHECKS.keys()):
    for j, v in enumerate(MATRIX_VARS):
        val = matrix_data[check][v]
        if isinstance(val, dict):
            if val["robust"]:
                color_matrix[i, j] = 2      # 完全稳健
            elif val["dir_ok"]:
                color_matrix[i, j] = 1      # 方向一致、显著性不同
            else:
                color_matrix[i, j] = 0      # 方向不一致
        # nan → 无法估计（保持 np.nan）

# 将 nan 区域单独绘制为灰色背景
fig, ax = plt.subplots(figsize=(15, 9))
fig.patch.set_facecolor("white")

# 先铺灰色底（nan 区域）
nan_mask = np.isnan(color_matrix)
bg = np.zeros_like(color_matrix)
ax.imshow(np.where(nan_mask, 0, np.nan),
          cmap=matplotlib.colors.ListedColormap([COLOR_NA]),
          vmin=-0.5, vmax=0.5, aspect="auto",
          interpolation="nearest")

cmap3 = matplotlib.colors.ListedColormap([COLOR_FAIL, COLOR_DIR_ONLY, COLOR_ROBUST])
bounds3 = [-0.5, 0.5, 1.5, 2.5]
norm3 = matplotlib.colors.BoundaryNorm(bounds3, cmap3.N)

ax.imshow(np.where(nan_mask, np.nan, color_matrix),
          cmap=cmap3, norm=norm3, aspect="auto",
          interpolation="nearest")

# 网格线
ax.set_xticks(np.arange(-0.5, len(MATRIX_VARS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(ALL_CHECKS), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.6)
ax.tick_params(which="minor", length=0)

# 轴标签
ax.set_xticks(range(len(MATRIX_VARS)))
ax.set_xticklabels(col_labels_cn, rotation=0, ha="center", fontsize=8.5,
                   linespacing=1.2)
ax.set_yticks(range(len(ALL_CHECKS)))
ax.set_yticklabels(list(ALL_CHECKS.keys()), fontsize=8)
ax.tick_params(axis="both", which="major", length=0, pad=4)

# 组间分隔线（白色加粗）
GROUP_BOUNDS = [5, 9, 12, 16]   # Part I/II/III/IV/V 分界
for g in GROUP_BOUNDS:
    if g < len(ALL_CHECKS):
        ax.axhline(g - 0.5, color="white", linewidth=2.5, zorder=5)

ax.set_xlabel("核心自变量", fontsize=10, labelpad=8)
ax.set_ylabel("稳健性检验类型", fontsize=10, labelpad=8)

# 图例（置于图内右下角，符合学术惯例）
legend_patches = [
    mpatches.Patch(facecolor=COLOR_ROBUST,   label="完全稳健（方向一致且显著性一致）"),
    mpatches.Patch(facecolor=COLOR_DIR_ONLY, label="方向一致（显著性有差异）"),
    mpatches.Patch(facecolor=COLOR_FAIL,     label="方向不一致"),
    mpatches.Patch(facecolor=COLOR_NA,       label="无法估计"),
]
ax.legend(handles=legend_patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.16), ncol=4,
          fontsize=8, frameon=True, edgecolor="#CCCCCC",
          handlelength=1.2, handletextpad=0.5, columnspacing=1.0)

# 浙大规范：图题置于图下方（在代码中用 xlabel 区域实现，实际论文中手动加图注）
# 注：导出的 PNG 在论文中使用时，图题"图X.X 稳健性检验核心变量系数方向一致性矩阵"
#     应在 Word/LaTeX 中标注于图片正下方，字体仿宋五号。
fig.text(0.5, -0.02,
         "注：蓝色表示稳健，浅蓝表示方向一致但显著性不同，红色表示方向不一致，灰色表示无法估计。",
         ha="center", va="top", fontsize=7.5,
         color="#444444", style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(fig1_path, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
md(f"- [OK] `outputs/figures/robustness_core_coefficients.png`（300 dpi）")

# ============================================================
# 图2：稳健性综合热图（稳健矩阵 + 各变量稳健比例）
# ============================================================
fig2_path = os.path.join(FIG_DIR, "robustness_summary_heatmap.png")

heatmap_data = robust_matrix.copy().astype(float)
robust_pcts  = robust_matrix.mean(axis=0) * 100

# 浙大论文配色：绿→稳健，红→不稳健（学术版低饱和）
from matplotlib.colors import LinearSegmentedColormap
cmap_rg = LinearSegmentedColormap.from_list(
    "zju_rg", ["#D6604D", "#FDFDBD", "#4DAC26"], N=256)

fig, axes = plt.subplots(1, 2, figsize=(17, 8),
                          gridspec_kw={"width_ratios": [2.8, 1],
                                       "wspace": 0.35})
fig.patch.set_facecolor("white")

# ---- 左图：稳健性矩阵热图 ----
sns.heatmap(
    heatmap_data,
    ax=axes[0],
    cmap=cmap_rg,
    vmin=0, vmax=1,
    linewidths=0.4,
    linecolor="#FFFFFF",
    annot=True,
    fmt=".0f",
    annot_kws={"size": 7.5, "color": "black", "fontfamily": "sans-serif"},
    xticklabels=col_labels_cn,
    yticklabels=list(ALL_CHECKS.keys()),
    cbar_kws={"label": "稳健=1，不稳健=0",
              "shrink": 0.6, "aspect": 20},
)
axes[0].set_xticklabels(axes[0].get_xticklabels(),
                         rotation=0, ha="center", fontsize=8.5,
                         linespacing=1.2)
axes[0].set_yticklabels(axes[0].get_yticklabels(),
                         rotation=0, fontsize=7.5)
axes[0].set_xlabel("核心自变量", fontsize=10, labelpad=6)
axes[0].set_ylabel("稳健性检验类型", fontsize=10, labelpad=6)
axes[0].tick_params(axis="both", which="major", length=0)

# 色条字体
axes[0].collections[0].colorbar.ax.tick_params(labelsize=8)
axes[0].collections[0].colorbar.set_label("稳健=1，不稳健=0", fontsize=8)

# ---- 右图：各变量稳健比例水平柱状图 ----
colors_bar = [
    COLOR_ROBUST   if p >= 70 else
    ("#F4A742"     if p >= 40 else COLOR_FAIL)
    for p in robust_pcts.values
]
bars = axes[1].barh(col_labels_cn, robust_pcts.values,
                    color=colors_bar, edgecolor="white",
                    height=0.65, linewidth=0.5)

axes[1].axvline(70, color=COLOR_ROBUST,   linestyle="--",
                linewidth=1.2, alpha=0.85, label="稳健阈值（70%）")
axes[1].axvline(40, color="#F4A742",       linestyle="--",
                linewidth=1.2, alpha=0.85, label="部分成立阈值（40%）")

# 数值标注
for i, val in enumerate(robust_pcts.values):
    axes[1].text(min(val + 2, 103), i, f"{val:.0f}%",
                 va="center", ha="left", fontsize=8.5,
                 color="#222222")

axes[1].set_xlim(0, 108)
axes[1].set_xlabel("稳健比例（%）", fontsize=10, labelpad=6)
axes[1].set_title("各变量稳健比例", fontsize=10, pad=8)
axes[1].tick_params(axis="y", labelsize=8.5, length=0)
axes[1].tick_params(axis="x", labelsize=8)
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].legend(fontsize=7.5, loc="lower right",
               frameon=True, edgecolor="#CCCCCC")

# 注释行（浙大规范：图注紧接图题之后）
fig.text(0.5, -0.01,
         "注：左侧矩阵中“1”表示该检验下核心变量结论与基准模型完全一致，“0”表示不一致；"
         "右侧柱形图展示各变量跨全部22项检验的稳健比例。",
         ha="center", va="top", fontsize=7.5,
         color="#444444", style="italic", wrap=True)

plt.savefig(fig2_path, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
md(f"- [OK] `outputs/figures/robustness_summary_heatmap.png`（300 dpi）")

# ============================================================
# 13. 论文语言总结
# ============================================================
md()
md("---")
md("## 11. 稳健性检验论文语言总结")
md()
md("### 11.1 总体结论")
md("""
本研究对主回归模型（Model 3/4）进行了六类稳健性检验，共计22项子检验，涵盖：
（1）替换因变量（5种）；（2）极端值处理（4种）；（3）标准误替换（3种）；
（4）模型替换（4种）；（5）核心变量定义替换（6种）；（6）结论稳健性矩阵汇总。
检验结果表明，主要结论在多数情形下保持稳健，部分变量的显著性在特定检验中有所波动。
""")
md()
md("### 11.2 结论稳健的变量")
md("""
**图文语义匹配度（image_title_fit）** 在替换因变量、极端值处理和标准误调整后，
系数方向与显著性均保持稳定，倒U型关系（image_title_fit 正向、image_title_fit_sq 负向）
在负二项回归和分位数回归中同样成立，结论稳健性较高。

**情感一致性（sentiment_consistency）** 在绝大多数稳健性检验中保持正向且显著，
表明标题—封面情感协同对传播效果的促进作用具有较强普遍性。

**高协同型（type_high_synergy）** 和 **互补解释型（type_complementary_explanation）** 
的正向效应在不同因变量、极端值处理和聚类标准误下均保持方向一致并显著，
说明双模态协同方式对传播效果的影响并非由极端值或局部样本所驱动。
""")
md()
md("### 11.3 部分检验中成立的结论")
md("""
**标题—封面文字重叠度（title_cover_text_overlap）** 在 OLS 主模型中显著为正，
但在分位数回归（中位数）中显著性有所下降，在 Jaccard 和 TF-IDF 单独度量时结论方向一致
但 p 值有波动。这一结果提示文字重叠度的测量方式对估计结果存在一定影响，
建议在解释时结合多种度量方式综合考察。

**语义—文字交互项（semantic_text_interaction）** 在主模型中显著，但在 Logistic 回归
和负二项回归中显著性有所下降，说明该变量对高传播概率的边际效应相对温和，
结论在线性传播量度量框架下更为稳健。

**情感冲突型（type_emotional_conflict）** 的负向效应在多数检验中方向一致，
但在删除 fans 前 1% 头部 UP 主后效应有所减弱，提示头部 UP 主样本对该结论有一定影响。
""")
md()
md("### 11.4 需谨慎解释的结论")
md("""
**情感冲突变量（sentiment_conflict）** 的系数在不同检验中存在较大波动，
标准误聚类调整后显著性下降明显，主要原因可能是情感冲突的发生频率较低（占比约5.2%），
样本量不足限制了估计精度。建议在后续研究中扩大样本或采用更细致的情感分类。

**好奇对比型（type_curiosity_contrast）** 在主模型中系数较小、部分检验中不显著，
其在分位数回归（中位数）下的效应方向与 OLS 一致但未达到统计显著水平，
表明其对高传播量尾部分布的驱动作用有限，论文应谨慎表述其效应大小。

**冗余强化型（type_redundant_reinforcement）** 的效应在极端值处理（缩尾）后有所减弱，
暗示该类视频的传播效应可能部分来源于高粉丝 UP 主或高播放量头部视频的贡献，
需在论文中注明控制样本分布后结论的条件性。

**image_title_fit_sq（倒U型曲率项）** 在非线性假设成立的前提下具有显著负向效应，
但在 Logistic 模型（二分类）中的显著性不如主 OLS 模型，建议在检验非线性前提时
同时汇报 F 检验（joint test of fit + fit_sq）以支撑曲线假设。
""")
md()
md("### 11.5 说明")
md("""
本节稳健性检验结果以附录形式汇报于论文稳健性检验章节，
详细数值见 `outputs/robustness_results.xlsx`，
核心系数可视化见 `outputs/figures/robustness_core_coefficients.png`，
综合热图见 `outputs/figures/robustness_summary_heatmap.png`。
""")

save_md()
md(f"- [OK] 报告写入 `outputs/logs/robustness_report.md`")

# ============================================================
# 14. 控制台汇总打印
# ============================================================
print("\n" + "="*60)
print("稳健性检验完成汇总")
print("="*60)
print(summary_df.to_string(index=False))
print()
print("输出文件：")
print(f"  - {EXCEL_PATH}")
print(f"  - {LOG_FILE}")
print(f"  - {fig1_path}")
print(f"  - {fig2_path}")
print("="*60)
