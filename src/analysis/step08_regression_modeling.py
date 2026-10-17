#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第八步：递进式回归建模
研究题目：标题和封面双模态协同对社交媒体平台视频传播的影响——以B站知识类视频为例
因变量：log_view, log_view_per_day
聚类稳健标准误：按 uid 聚类

依赖包：
    pip install pandas numpy statsmodels openpyxl matplotlib seaborn scipy python-docx
"""

import os
import sys
import traceback
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 0. 基础配置
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings('ignore')

# 中文字体配置（优先使用系统可用字体）
def configure_chinese_font():
    """配置 matplotlib 中文字体"""
    import matplotlib.font_manager as fm
    font_candidates = [
        'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei',
        'STHeiti', 'Microsoft YaHei', 'Arial Unicode MS',
        'Hiragino Sans GB', 'DejaVu Sans'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in font_candidates:
        if font in available:
            rcParams['font.family'] = font
            break
    else:
        rcParams['font.family'] = 'DejaVu Sans'
    rcParams['axes.unicode_minus'] = False

configure_chinese_font()

# 路径配置
INPUT_FILE  = "outputs/modeling_dataset.csv"
OUT_DIR     = "outputs"
FIG_DIR     = os.path.join(OUT_DIR, "figures")
LOG_DIR     = os.path.join(OUT_DIR, "logs")

for d in [OUT_DIR, FIG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# 日志配置
log_path = os.path.join(LOG_DIR, "regression_model_report.md")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "regression_run.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 变量定义字典（英文变量名 → 中文别名）
# ─────────────────────────────────────────────────────────────────────────────
VAR_LABELS = {
    # 因变量
    'log_view':              '对数播放量',
    'log_view_per_day':      '对数日均播放量',
    # 控制变量
    'log_fans':              '对数粉丝量',
    'log_age':               '对数视频在线天数',
    'log_duration':          '对数视频时长',
    'tag_count':             '标签数量',
    'description_len':       '简介长度',
    'is_collection':         '是否合集',
    'copyright':             '版权类型',
    'C(domain)':             '知识领域（分类）',
    'C(level)':              '用户等级（分类）',
    'C(pub_month)':          '发布月份（分类）',
    'C(pub_weekday)':        '发布星期（分类）',
    'C(pub_hour)':           '发布小时（分类）',
    # 标题特征
    'title_char_len':             '标题字符长度',
    'title_question_type':        '标题疑问句类型',
    'title_exclamation_type':     '标题感叹句类型',
    'title_has_number':           '标题含数字',
    'title_sentiment_intensity':  '标题情感强度',
    'curiosity_marker_count':     '好奇心标记数',
    'practical_value_word_count': '实用价值词频',
    # 封面特征
    'image_brightness':           '封面亮度',
    'image_contrast':             '封面对比度',
    'image_colorfulness':         '封面色彩丰富度',
    'visual_complexity':          '视觉复杂度',
    'cover_has_text':             '封面含文字',
    'cover_text_area_ratio':      '封面文字面积比',
    'face_presence':              '封面人脸存在',
    'cover_has_chart_or_formula': '封面含图表/公式',
    'cover_is_slide_or_screenshot':'封面为截图/幻灯片',
    # 双模态协同
    'image_title_fit_centered':     '标题-封面语义相似度（中心化）',
    'image_title_fit_sq_centered':  '标题-封面语义相似度平方（中心化）',
    'title_cover_text_overlap':     '标题-封面文字重叠度',
    'sentiment_consistency':        '情感一致性',
    'sentiment_conflict':           '情感冲突',
    'C(complementarity_type)':      '互补类型（分类）',
    # 交互效应
    'semantic_text_interaction':    '语义文本交互项',
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. 读取数据
# ─────────────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("第八步：递进式回归建模 开始")
logger.info(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 60)

if not os.path.exists(INPUT_FILE):
    logger.error(f"输入文件不存在：{INPUT_FILE}")
    sys.exit(1)

df_raw = pd.read_csv(INPUT_FILE, low_memory=False)
logger.info(f"读取数据：{df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 变量分组定义
# ─────────────────────────────────────────────────────────────────────────────
DEPVARS = ['log_view', 'log_view_per_day']

CTRL_CONTINUOUS = [
    'log_fans', 'log_age', 'log_duration',
    'tag_count', 'description_len'
]
CTRL_BINARY = ['is_collection', 'copyright']
CTRL_CATEG  = ['domain', 'level', 'pub_month', 'pub_weekday', 'pub_hour']

TITLE_VARS = [
    'title_char_len', 'title_question_type', 'title_exclamation_type',
    'title_has_number', 'title_sentiment_intensity',
    'curiosity_marker_count', 'practical_value_word_count'
]

COVER_VARS = [
    'image_brightness', 'image_contrast', 'image_colorfulness',
    'visual_complexity', 'cover_has_text', 'cover_text_area_ratio',
    'face_presence', 'cover_has_chart_or_formula', 'cover_is_slide_or_screenshot'
]

MODAL_VARS_BASE = [
    'image_title_fit_centered', 'image_title_fit_sq_centered',
    'title_cover_text_overlap', 'sentiment_consistency', 'sentiment_conflict'
]
MODAL_CATEG = ['complementarity_type']

INTERACTION_VARS = ['semantic_text_interaction']

ALL_REQUIRED_VARS = (
    DEPVARS + ['uid'] + CTRL_CONTINUOUS + CTRL_BINARY + CTRL_CATEG +
    TITLE_VARS + COVER_VARS + MODAL_VARS_BASE + MODAL_CATEG + INTERACTION_VARS
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. 缺失变量检查（分组报告）
# ─────────────────────────────────────────────────────────────────────────────
def check_missing_vars(df, var_groups: dict) -> dict:
    """检查各组变量缺失情况，返回缺失变量字典"""
    missing = {}
    for group_name, var_list in var_groups.items():
        miss = [v for v in var_list if v not in df.columns]
        if miss:
            missing[group_name] = miss
            logger.warning(f"【缺失变量】{group_name} 组缺失：{miss}")
        else:
            logger.info(f"【变量检查】{group_name} 组全部存在 ✓")
    return missing

var_groups_check = {
    '因变量':        DEPVARS,
    '聚类变量':      ['uid'],
    '控制变量（连续）': CTRL_CONTINUOUS,
    '控制变量（二值）': CTRL_BINARY,
    '控制变量（分类）': CTRL_CATEG,
    '标题特征变量':   TITLE_VARS,
    '封面特征变量':   COVER_VARS,
    '双模态协同变量': MODAL_VARS_BASE + MODAL_CATEG,
    '交互效应变量':   INTERACTION_VARS,
}

missing_report = check_missing_vars(df_raw, var_groups_check)
if missing_report:
    logger.error("存在缺失变量组，请检查上游步骤输出。脚本将跳过含缺失变量的模型。")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 数据清洗与预处理
# ─────────────────────────────────────────────────────────────────────────────
df = df_raw.copy()

# 5.1 分类变量类型转换 & 参照组设置
CATEG_REF = {
    'domain':      '科学科普',   # 参照：科学科普
    'level':       '1',          # 参照：等级1（最低等级）
    'pub_month':   '1',          # 参照：1月
    'pub_weekday': '1',          # 参照：周一
    'pub_hour':    '0',          # 参照：0时
    'complementarity_type': 'neutral',  # 参照：中性互补
}

for col in CTRL_CATEG + MODAL_CATEG:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        if col in CATEG_REF and CATEG_REF[col] in df[col].unique():
            cats = [CATEG_REF[col]] + sorted(
                [c for c in df[col].unique() if c != CATEG_REF[col]]
            )
            df[col] = pd.Categorical(df[col], categories=cats, ordered=False)
            logger.info(f"分类变量 {col} 参照组设为：{CATEG_REF[col]}")
        else:
            df[col] = pd.Categorical(df[col])
            logger.info(f"分类变量 {col} 使用默认参照组：{df[col].cat.categories[0]}")

# 5.2 连续核心解释变量标准化（z-score）
VARS_TO_STANDARDIZE = (
    CTRL_CONTINUOUS + TITLE_VARS + COVER_VARS + MODAL_VARS_BASE + INTERACTION_VARS
)

std_suffix = '_std'
standardized_map = {}

for v in VARS_TO_STANDARDIZE:
    if v not in df.columns:
        continue
    col_std = v + std_suffix
    col_data = pd.to_numeric(df[v], errors='coerce')
    mu, sigma = col_data.mean(), col_data.std()
    if sigma > 0:
        df[col_std] = (col_data - mu) / sigma
    else:
        df[col_std] = 0.0
        logger.warning(f"变量 {v} 标准差为0，标准化后全为0")
    standardized_map[v] = col_std

logger.info(f"已标准化 {len(standardized_map)} 个连续变量（均值0，标准差1）")

# 5.3 image_title_fit 平方项（使用标准化后的值）
if 'image_title_fit_centered_std' in df.columns:
    df['image_title_fit_sq_centered_std'] = df['image_title_fit_centered_std'] ** 2

# 5.4 缺失值处理
required_for_analysis = (
    DEPVARS + ['uid'] + CTRL_CONTINUOUS + CTRL_BINARY
)
available_required = [c for c in required_for_analysis if c in df.columns]
df_model = df.dropna(subset=available_required).copy()
n_dropped = len(df) - len(df_model)
logger.info(f"删除核心变量缺失行：{n_dropped} 行，剩余 {len(df_model)} 行")

if n_dropped > 0:
    logger.warning(f"共删除 {n_dropped} 行含核心变量缺失值的观测")

# 5.5 异常值检查（IQR 方法，仅记录日志，不删除）
for v in DEPVARS:
    if v in df_model.columns:
        q1, q3 = df_model[v].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = ((df_model[v] < q1 - 3*iqr) | (df_model[v] > q3 + 3*iqr)).sum()
        logger.info(f"因变量 {v}：IQR3×法检测到 {n_out} 个极端值（未删除）")

logger.info(f"最终建模数据集：{df_model.shape[0]} 行 × {df_model.shape[1]} 列")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 公式构建辅助函数
# ─────────────────────────────────────────────────────────────────────────────
def _get_std(v, standardized_map):
    """返回标准化变量名（若存在），否则返回原始变量名"""
    return standardized_map.get(v, v)

def build_formula(depvar, var_blocks, df, standardized_map):
    """
    构建 statsmodels formula 字符串，自动过滤不存在的变量。
    返回：(formula_str, missing_vars_list)
    """
    terms = []
    missing = []
    for term in var_blocks:
        # 检测是否为分类项 C(xxx)
        if term.startswith('C(') and term.endswith(')'):
            col = term[2:-1]
            if col in df.columns:
                terms.append(term)
            else:
                missing.append(term)
        # 交互项 a:b 或 a*b
        elif ':' in term or ('*' in term and 'C(' not in term):
            parts = term.replace('*', ':').split(':')
            resolved = []
            ok = True
            for p in parts:
                p = p.strip()
                std_p = _get_std(p, standardized_map)
                if std_p in df.columns:
                    resolved.append(std_p)
                elif p in df.columns:
                    resolved.append(p)
                else:
                    ok = False
                    missing.append(p)
            if ok:
                sep = ':' if ':' in term else '*'
                terms.append(sep.join(resolved))
        # 含C()的交互项
        elif '*' in term and 'C(' in term:
            # e.g. image_title_fit_centered_std * C(domain)
            parts = [p.strip() for p in term.split('*')]
            resolved = []
            ok = True
            for p in parts:
                if p.startswith('C(') and p.endswith(')'):
                    col = p[2:-1]
                    if col in df.columns:
                        resolved.append(p)
                    else:
                        ok = False
                        missing.append(p)
                else:
                    std_p = _get_std(p, standardized_map)
                    if std_p in df.columns:
                        resolved.append(std_p)
                    elif p in df.columns:
                        resolved.append(p)
                    else:
                        ok = False
                        missing.append(p)
            if ok:
                terms.append(' * '.join(resolved))
        # 普通连续/二值变量
        else:
            std_v = _get_std(term, standardized_map)
            if std_v in df.columns:
                terms.append(std_v)
            elif term in df.columns:
                terms.append(term)
            else:
                missing.append(term)

    formula = f"{depvar} ~ " + " + ".join(terms) if terms else None
    return formula, missing

# ─────────────────────────────────────────────────────────────────────────────
# 7. 模型变量块定义
# ─────────────────────────────────────────────────────────────────────────────

# 控制变量块（原始变量名，build_formula 会自动替换为标准化版）
CTRL_BLOCK = (
    CTRL_CONTINUOUS +
    CTRL_BINARY +
    ['C(domain)', 'C(level)', 'C(pub_month)', 'C(pub_weekday)', 'C(pub_hour)']
)

TITLE_BLOCK = TITLE_VARS

COVER_BLOCK = COVER_VARS

MODAL_BLOCK = (
    MODAL_VARS_BASE +
    ['C(complementarity_type)']
)

INTERACTION_BLOCK = (
    INTERACTION_VARS +
    ['image_title_fit_centered_std * C(domain)',
     'sentiment_consistency * C(domain)']
)

MODEL_BLOCKS = {
    'Model 0': CTRL_BLOCK,
    'Model 1': CTRL_BLOCK + TITLE_BLOCK,
    'Model 2': CTRL_BLOCK + TITLE_BLOCK + COVER_BLOCK,
    'Model 3': CTRL_BLOCK + TITLE_BLOCK + COVER_BLOCK + MODAL_BLOCK,
    'Model 4': CTRL_BLOCK + TITLE_BLOCK + COVER_BLOCK + MODAL_BLOCK + INTERACTION_BLOCK,
}

MODEL_NAMES_CN = {
    'Model 0': '模型0：控制变量',
    'Model 1': '模型1：+标题特征',
    'Model 2': '模型2：+封面特征',
    'Model 3': '模型3：+双模态协同',
    'Model 4': '模型4：+交互效应',
}

# ─────────────────────────────────────────────────────────────────────────────
# 8. 聚类稳健标准误回归函数
# ─────────────────────────────────────────────────────────────────────────────
def run_cluster_ols(formula, df, cluster_col='uid'):
    """
    使用 statsmodels 运行 OLS 回归，返回按 cluster_col 聚类稳健标准误结果。
    """
    try:
        model = smf.ols(formula=formula, data=df)
        # groups 必须与观测一一对应
        groups = df[cluster_col].values
        result = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
        return result
    except Exception as e:
        logger.error(f"模型拟合失败：{e}")
        logger.error(traceback.format_exc())
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 9. 提取回归结果
# ─────────────────────────────────────────────────────────────────────────────
def extract_results(result, model_name):
    """
    从 statsmodels 回归结果中提取系数表、拟合统计量。
    返回：coef_df, summary_dict
    """
    if result is None:
        return None, None

    coef_df = pd.DataFrame({
        'variable':    result.params.index,
        'coefficient': result.params.values,
        'std_err':     result.bse.values,
        't_value':     result.tvalues.values,
        'p_value':     result.pvalues.values,
        'ci_lower':    result.conf_int().iloc[:, 0].values,
        'ci_upper':    result.conf_int().iloc[:, 1].values,
    })

    def sig_stars(p):
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        elif p < 0.1:  return '†'
        return ''

    coef_df['significance'] = coef_df['p_value'].apply(sig_stars)
    coef_df['model'] = model_name

    summary = {
        'model':       model_name,
        'N':           int(result.nobs),
        'R2':          result.rsquared,
        'R2_adj':      result.rsquared_adj,
        'AIC':         result.aic,
        'BIC':         result.bic,
        'F_stat':      getattr(result, 'fvalue', np.nan),
        'F_pvalue':    getattr(result, 'f_pvalue', np.nan),
        'df_model':    result.df_model,
        'df_resid':    result.df_resid,
    }

    return coef_df, summary

# ─────────────────────────────────────────────────────────────────────────────
# 10. 主回归循环
# ─────────────────────────────────────────────────────────────────────────────
all_results   = {}   # {depvar: {model_name: result_obj}}
all_coef_dfs  = {}   # {depvar: {model_name: coef_df}}
all_summaries = {}   # {depvar: {model_name: summary_dict}}
formula_log   = {}   # 记录每个模型实际使用的公式

for depvar in DEPVARS:
    if depvar not in df_model.columns:
        logger.error(f"因变量 {depvar} 不在数据中，跳过")
        continue

    logger.info(f"\n{'='*50}")
    logger.info(f"因变量：{depvar} ({VAR_LABELS.get(depvar, depvar)})")
    logger.info(f"{'='*50}")

    all_results[depvar]   = {}
    all_coef_dfs[depvar]  = {}
    all_summaries[depvar] = {}
    formula_log[depvar]   = {}

    for model_name, var_block in MODEL_BLOCKS.items():
        logger.info(f"\n  >> {model_name} ({MODEL_NAMES_CN[model_name]})")

        formula, missing_vars = build_formula(
            depvar, var_block, df_model, standardized_map
        )

        if missing_vars:
            logger.warning(f"     跳过/降级，以下变量缺失：{missing_vars}")

        if formula is None:
            logger.error(f"     {model_name} 无法构建公式，跳过")
            continue

        logger.info(f"     公式：{formula[:200]}{'...' if len(formula)>200 else ''}")
        formula_log[depvar][model_name] = formula

        # 拟合模型
        result = run_cluster_ols(formula, df_model, cluster_col='uid')

        if result is not None:
            coef_df, summary = extract_results(result, model_name)
            all_results[depvar][model_name]   = result
            all_coef_dfs[depvar][model_name]  = coef_df
            all_summaries[depvar][model_name] = summary

            logger.info(f"     N={summary['N']:,}  R²={summary['R2']:.4f}  "
                        f"Adj.R²={summary['R2_adj']:.4f}  "
                        f"AIC={summary['AIC']:.1f}  BIC={summary['BIC']:.1f}")
        else:
            logger.error(f"     {model_name} 拟合失败，已跳过")

# ─────────────────────────────────────────────────────────────────────────────
# 11. 构建汇总回归表（论文格式）
# ─────────────────────────────────────────────────────────────────────────────
def build_regression_table(depvar, all_coef_dfs, all_summaries, model_order):
    """
    构建宽格式回归汇总表，格式：变量行 × 模型列。
    每个单元格格式：系数 + 显著性星号（换行）标准误（括号内）
    """
    model_names_avail = [m for m in model_order if m in all_coef_dfs.get(depvar, {})]
    if not model_names_avail:
        return None, None

    # 收集所有变量
    all_vars = []
    seen = set()
    for mn in model_names_avail:
        for v in all_coef_dfs[depvar][mn]['variable']:
            if v not in seen and v != 'Intercept':
                all_vars.append(v)
                seen.add(v)

    rows = []
    for v in all_vars:
        row = {'variable': v, 'cn_label': VAR_LABELS.get(v, v)}
        for mn in model_names_avail:
            cdf = all_coef_dfs[depvar][mn]
            match = cdf[cdf['variable'] == v]
            if not match.empty:
                coef = match['coefficient'].values[0]
                se   = match['std_err'].values[0]
                sig  = match['significance'].values[0]
                row[f'{mn}_coef'] = f"{coef:.3f}{sig}"
                row[f'{mn}_se']   = f"({se:.3f})"
            else:
                row[f'{mn}_coef'] = ''
                row[f'{mn}_se']   = ''
        rows.append(row)

    # 截距行
    row_int = {'variable': 'Intercept', 'cn_label': '截距'}
    for mn in model_names_avail:
        cdf = all_coef_dfs[depvar][mn]
        match = cdf[cdf['variable'] == 'Intercept']
        if not match.empty:
            coef = match['coefficient'].values[0]
            se   = match['std_err'].values[0]
            sig  = match['significance'].values[0]
            row_int[f'{mn}_coef'] = f"{coef:.3f}{sig}"
            row_int[f'{mn}_se']   = f"({se:.3f})"
        else:
            row_int[f'{mn}_coef'] = ''
            row_int[f'{mn}_se']   = ''
    rows.append(row_int)

    # 模型统计行
    stat_rows = []
    for stat_label, stat_key, fmt in [
        ('N（观测数）',    'N',     '{:,}'),
        ('R²',            'R2',    '{:.4f}'),
        ('调整R²',        'R2_adj','{:.4f}'),
        ('AIC',           'AIC',   '{:.1f}'),
        ('BIC',           'BIC',   '{:.1f}'),
    ]:
        row_s = {'variable': stat_key, 'cn_label': stat_label}
        for mn in model_names_avail:
            s = all_summaries.get(depvar, {}).get(mn, {})
            val = s.get(stat_key, np.nan)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                row_s[f'{mn}_coef'] = fmt.format(val)
            else:
                row_s[f'{mn}_coef'] = ''
            row_s[f'{mn}_se'] = ''
        stat_rows.append(row_s)

    coef_df_table = pd.DataFrame(rows)
    stat_df_table = pd.DataFrame(stat_rows)

    return coef_df_table, stat_df_table, model_names_avail

# ─────────────────────────────────────────────────────────────────────────────
# 12. 输出 Excel 回归结果表
# ─────────────────────────────────────────────────────────────────────────────
def write_excel_results(filepath, all_coef_dfs, all_summaries, all_results, model_order):
    """写出 Excel 多 Sheet 回归结果"""
    from openpyxl import Workbook
    from openpyxl.styles import (
        Font, Alignment, Border, Side, PatternFill, numbers
    )
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)  # 移除默认空白Sheet

    thin = Side(style='thin', color='000000')
    border = Border(top=thin, bottom=thin, left=thin, right=thin)

    header_fill   = PatternFill("solid", fgColor="1F4E79")
    subheader_fill= PatternFill("solid", fgColor="2E75B6")
    alt_fill      = PatternFill("solid", fgColor="EBF3FB")
    stat_fill     = PatternFill("solid", fgColor="D6E4F0")

    def style_header(cell, level=1):
        cell.font      = Font(bold=True, color="FFFFFF",
                              size=11 if level==1 else 10)
        cell.fill      = header_fill if level==1 else subheader_fill
        cell.alignment = Alignment(horizontal='center', vertical='center',
                                   wrap_text=True)
        cell.border    = border

    def style_cell(cell, alt=False, bold=False):
        cell.alignment = Alignment(horizontal='center', vertical='center',
                                   wrap_text=True)
        cell.border    = border
        if bold:
            cell.font = Font(bold=True, size=10)
        if alt:
            cell.fill = alt_fill

    # ── Sheet 1：变量说明表 ──────────────────────────────────────────────────
    ws_var = wb.create_sheet("变量说明表")
    headers = ['变量名（英文）', '中文别名', '变量类型', '描述', '标准化']
    var_type_map = {
        **{v: '因变量（连续）' for v in DEPVARS},
        **{v: '控制变量（连续）' for v in CTRL_CONTINUOUS},
        **{v: '控制变量（二值）' for v in CTRL_BINARY},
        **{v: '控制变量（分类）' for v in CTRL_CATEG},
        **{v: '标题特征（连续/二值）' for v in TITLE_VARS},
        **{v: '封面特征（连续/二值）' for v in COVER_VARS},
        **{v: '双模态协同（连续）' for v in MODAL_VARS_BASE},
        **{v: '双模态协同（分类）' for v in MODAL_CATEG},
        **{v: '交互效应' for v in INTERACTION_VARS},
    }
    for c, h in enumerate(headers, 1):
        cell = ws_var.cell(row=1, column=c, value=h)
        style_header(cell)

    all_vars_desc = (DEPVARS + ['uid'] + CTRL_CONTINUOUS + CTRL_BINARY +
                     CTRL_CATEG + TITLE_VARS + COVER_VARS +
                     MODAL_VARS_BASE + MODAL_CATEG + INTERACTION_VARS)
    for r, v in enumerate(all_vars_desc, 2):
        is_std = v in standardized_map
        vals = [
            v, VAR_LABELS.get(v, v), var_type_map.get(v, '其他'),
            '', '是' if is_std else '否'
        ]
        for c, val in enumerate(vals, 1):
            cell = ws_var.cell(row=r, column=c, value=val)
            style_cell(cell, alt=(r % 2 == 0))

    ws_var.column_dimensions['A'].width = 38
    ws_var.column_dimensions['B'].width = 25
    ws_var.column_dimensions['C'].width = 22
    ws_var.column_dimensions['D'].width = 35
    ws_var.column_dimensions['E'].width = 10

    # ── Sheet 2 & 3：完整系数表（每个因变量一个Sheet）────────────────────────
    for depvar in DEPVARS:
        ws = wb.create_sheet(f"系数表_{depvar}")
        model_names_avail = [m for m in model_order
                             if m in all_coef_dfs.get(depvar, {})]
        if not model_names_avail:
            ws.cell(row=1, column=1, value="无可用模型结果")
            continue

        # 标题行
        ws.merge_cells(start_row=1, start_column=1,
                       end_row=1, end_column=2 + len(model_names_avail)*2)
        title_cell = ws.cell(row=1, column=1,
                             value=f"递进式回归结果（因变量：{VAR_LABELS.get(depvar,depvar)}）")
        title_cell.font      = Font(bold=True, size=13, color="FFFFFF")
        title_cell.fill      = header_fill
        title_cell.alignment = Alignment(horizontal='center', vertical='center')
        ws.row_dimensions[1].height = 28

        # 列标题
        ws.cell(row=2, column=1, value='变量名（英文）')
        ws.cell(row=2, column=2, value='中文别名')
        style_header(ws.cell(row=2, column=1))
        style_header(ws.cell(row=2, column=2))

        for ci, mn in enumerate(model_names_avail):
            col_base = 3 + ci * 2
            ws.merge_cells(start_row=2, start_column=col_base,
                           end_row=2, end_column=col_base+1)
            cell = ws.cell(row=2, column=col_base,
                           value=f"{mn}\n{MODEL_NAMES_CN.get(mn,'')}")
            style_header(cell)

        ws.cell(row=3, column=1, value='变量名')
        ws.cell(row=3, column=2, value='中文别名')
        style_header(ws.cell(row=3, column=1), level=2)
        style_header(ws.cell(row=3, column=2), level=2)
        for ci, _ in enumerate(model_names_avail):
            col_base = 3 + ci * 2
            c1 = ws.cell(row=3, column=col_base,     value='系数（标准误）')
            c2 = ws.cell(row=3, column=col_base + 1, value='t值 [p值]')
            style_header(c1, level=2)
            style_header(c2, level=2)

        # 收集变量
        all_vars_in_models = []
        seen_v = set()
        for mn in model_names_avail:
            for v in all_coef_dfs[depvar][mn]['variable']:
                if v not in seen_v:
                    all_vars_in_models.append(v)
                    seen_v.add(v)

        # 写数据行
        r = 4
        for vi, v in enumerate(all_vars_in_models):
            is_intercept = (v == 'Intercept')
            ws.cell(row=r, column=1, value=v)
            ws.cell(row=r, column=2, value=VAR_LABELS.get(v, v))
            style_cell(ws.cell(row=r, column=1), alt=(r%2==0), bold=is_intercept)
            style_cell(ws.cell(row=r, column=2), alt=(r%2==0))

            for ci, mn in enumerate(model_names_avail):
                col_base = 3 + ci * 2
                cdf = all_coef_dfs[depvar][mn]
                match = cdf[cdf['variable'] == v]
                if not match.empty:
                    coef  = match['coefficient'].values[0]
                    se    = match['std_err'].values[0]
                    tval  = match['t_value'].values[0]
                    pval  = match['p_value'].values[0]
                    sig   = match['significance'].values[0]
                    cell1 = ws.cell(row=r, column=col_base,
                                    value=f"{coef:.4f}{sig}\n({se:.4f})")
                    cell2 = ws.cell(row=r, column=col_base+1,
                                    value=f"{tval:.3f}\n[{pval:.4f}]")
                    for cell in [cell1, cell2]:
                        style_cell(cell, alt=(r%2==0))
                        cell.alignment = Alignment(horizontal='center',
                                                   vertical='center',
                                                   wrap_text=True)
                else:
                    for c_offset in [0, 1]:
                        cell = ws.cell(row=r, column=col_base+c_offset, value='—')
                        style_cell(cell, alt=(r%2==0))
            r += 1

        # 分隔行
        ws.cell(row=r, column=1, value='─── 模型统计 ───')
        for c in range(1, 3 + len(model_names_avail)*2):
            cell = ws.cell(row=r, column=c)
            cell.fill = stat_fill
            cell.font = Font(bold=True, size=9)
        r += 1

        # 统计行
        for stat_label, stat_key, fmt in [
            ('N（观测数）', 'N', '{:,}'),
            ('R²', 'R2', '{:.4f}'),
            ('调整R²', 'R2_adj', '{:.4f}'),
            ('AIC', 'AIC', '{:.2f}'),
            ('BIC', 'BIC', '{:.2f}'),
            ('ΔR²（相对Model0）', 'delta_R2', '{:.4f}'),
        ]:
            ws.cell(row=r, column=1, value=stat_key)
            ws.cell(row=r, column=2, value=stat_label)
            style_cell(ws.cell(row=r, column=1), bold=True)
            style_cell(ws.cell(row=r, column=2), bold=True)
            ws.cell(row=r, column=1).fill = stat_fill
            ws.cell(row=r, column=2).fill = stat_fill

            r2_m0 = all_summaries.get(depvar, {}).get('Model 0', {}).get('R2', np.nan)

            for ci, mn in enumerate(model_names_avail):
                col_base = 3 + ci * 2
                s = all_summaries.get(depvar, {}).get(mn, {})
                if stat_key == 'delta_R2':
                    r2_mn = s.get('R2', np.nan)
                    val = r2_mn - r2_m0 if not np.isnan(r2_mn) and not np.isnan(r2_m0) else np.nan
                else:
                    val = s.get(stat_key, np.nan)

                val_str = fmt.format(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else '—'
                ws.merge_cells(start_row=r, start_column=col_base,
                               end_row=r, end_column=col_base+1)
                cell = ws.cell(row=r, column=col_base, value=val_str)
                cell.fill = stat_fill
                cell.font = Font(bold=True, size=10)
                cell.alignment = Alignment(horizontal='center', vertical='center')
            r += 1

        # 注释行
        note = ("注：***p<0.001, **p<0.01, *p<0.05, †p<0.1；"
                "括号内为聚类稳健标准误（按UP主uid聚类）；"
                "所有连续变量已做z-score标准化；"
                f"分类变量参照组：domain={CATEG_REF.get('domain','')}, "
                f"level={CATEG_REF.get('level','')}")
        ws.cell(row=r+1, column=1, value=note)
        ws.merge_cells(start_row=r+1, start_column=1,
                       end_row=r+1, end_column=2+len(model_names_avail)*2)
        note_cell = ws.cell(row=r+1, column=1)
        note_cell.font      = Font(italic=True, size=9, color="555555")
        note_cell.alignment = Alignment(wrap_text=True, vertical='top')

        # 列宽
        ws.column_dimensions['A'].width = 42
        ws.column_dimensions['B'].width = 28
        for ci in range(len(model_names_avail)):
            col_base = 3 + ci * 2
            ws.column_dimensions[get_column_letter(col_base)].width = 18
            ws.column_dimensions[get_column_letter(col_base+1)].width = 16
        ws.row_dimensions[1].height = 30
        ws.row_dimensions[2].height = 38
        ws.row_dimensions[3].height = 32

    # ── Sheet 4：模型比较汇总 ────────────────────────────────────────────────
    ws_comp = wb.create_sheet("模型比较汇总")
    comp_headers = ['因变量', '模型', '模型说明', 'N', 'R²', '调整R²',
                    'ΔR²（vs Model0）', 'AIC', 'BIC']
    for c, h in enumerate(comp_headers, 1):
        cell = ws_comp.cell(row=1, column=c, value=h)
        style_header(cell)

    r = 2
    for depvar in DEPVARS:
        r2_m0 = all_summaries.get(depvar, {}).get('Model 0', {}).get('R2', np.nan)
        for mn in model_order:
            s = all_summaries.get(depvar, {}).get(mn, {})
            if not s:
                continue
            r2 = s.get('R2', np.nan)
            delta_r2 = r2 - r2_m0 if not np.isnan(r2) and not np.isnan(r2_m0) else np.nan
            row_vals = [
                VAR_LABELS.get(depvar, depvar), mn, MODEL_NAMES_CN.get(mn, ''),
                s.get('N', ''), f"{r2:.4f}" if not np.isnan(r2) else '—',
                f"{s.get('R2_adj', np.nan):.4f}",
                f"{delta_r2:.4f}" if not np.isnan(delta_r2) else '—',
                f"{s.get('AIC', np.nan):.2f}",
                f"{s.get('BIC', np.nan):.2f}",
            ]
            for c, val in enumerate(row_vals, 1):
                cell = ws_comp.cell(row=r, column=c, value=val)
                style_cell(cell, alt=(r%2==0))
            r += 1

    for col_i, width in enumerate([18,10,20,10,10,10,16,12,12], 1):
        ws_comp.column_dimensions[get_column_letter(col_i)].width = width

    wb.save(filepath)
    logger.info(f"Excel 结果已保存：{filepath}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. 输出 Word 回归报告
# ─────────────────────────────────────────────────────────────────────────────
def write_word_report(filepath, depvar, all_coef_dfs, all_summaries,
                      all_results, model_order, formula_log):
    """生成符合浙江大学论文格式的 Word 回归报告"""
    try:
        from docx import Document
        from docx.shared import Pt, Cm, RGBColor, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
        from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        import copy
    except ImportError:
        logger.error("python-docx 未安装，请运行：pip install python-docx")
        return

    doc = Document()

    # ── 页面设置（A4，2.5cm页边距）──────────────────────────────────────────
    for section in doc.sections:
        section.page_height   = Cm(29.7)
        section.page_width    = Cm(21.0)
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.5)

    # ── 样式辅助函数 ─────────────────────────────────────────────────────────
    def set_font(run, size=10.5, bold=False, italic=False, color=None):
        run.font.name  = '宋体'
        run.font.size  = Pt(size)
        run.font.bold  = bold
        run.font.italic = italic
        run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        if color:
            run.font.color.rgb = RGBColor(*color)

    def add_heading(doc, text, level=1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        sz = {1: 14, 2: 12, 3: 11}[level]
        set_font(run, size=sz, bold=True)
        pPr = p._element.get_or_add_pPr()
        spacing = OxmlElement('w:spacing')
        before = {1: '240', 2: '180', 3: '120'}[level]
        spacing.set(qn('w:before'), before)
        spacing.set(qn('w:after'), '120')
        pPr.append(spacing)
        return p

    def add_para(doc, text, size=10.5, bold=False, indent=False, align='left'):
        p = doc.add_paragraph()
        align_map = {
            'left':    WD_ALIGN_PARAGRAPH.LEFT,
            'center':  WD_ALIGN_PARAGRAPH.CENTER,
            'justify': WD_ALIGN_PARAGRAPH.JUSTIFY,
        }
        p.alignment = align_map.get(align, WD_ALIGN_PARAGRAPH.JUSTIFY)
        pf = p.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        if indent:
            pf.first_line_indent = Cm(0.74)
        run = p.add_run(text)
        set_font(run, size=size, bold=bold)
        return p

    def set_cell_style(cell, text, bold=False, size=9,
                       bg_color=None, align='center'):
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for p in cell.paragraphs:
            p.clear()
        p = cell.paragraphs[0]
        p.alignment = {
            'center': WD_ALIGN_PARAGRAPH.CENTER,
            'left':   WD_ALIGN_PARAGRAPH.LEFT,
            'right':  WD_ALIGN_PARAGRAPH.RIGHT,
        }.get(align, WD_ALIGN_PARAGRAPH.CENTER)
        run = p.add_run(str(text))
        set_font(run, size=size, bold=bold)
        if bg_color:
            tc_pr = cell._tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:val'), 'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), bg_color)
            tc_pr.append(shd)

    def set_table_borders(table):
        """给表格设置三线表样式（仅顶部双线、底部单线、表头底线）"""
        tbl = table._tbl
        tblPr = tbl.tblPr
        tblBorders = OxmlElement('w:tblBorders')
        for tag in ['top', 'bottom']:
            border = OxmlElement(f'w:{tag}')
            border.set(qn('w:val'), 'single')
            border.set(qn('w:sz'), '6')
            border.set(qn('w:space'), '0')
            border.set(qn('w:color'), '000000')
            tblBorders.append(border)
        for tag in ['left', 'right', 'insideH', 'insideV']:
            border = OxmlElement(f'w:{tag}')
            border.set(qn('w:val'), 'none')
            tblBorders.append(border)
        tblPr.append(tblBorders)

    # ── 封面页 ────────────────────────────────────────────────────────────────
    doc.add_paragraph()
    add_para(doc, '浙江大学硕士/博士学位论文', size=12, bold=False, align='center')
    doc.add_paragraph()
    add_para(doc,
             '标题和封面双模态协同对社交媒体平台视频传播的影响\n——以B站知识类视频为例',
             size=14, bold=True, align='center')
    doc.add_paragraph()
    add_para(doc,
             f'附录：递进式回归分析结果\n（因变量：{VAR_LABELS.get(depvar, depvar)}）',
             size=12, align='center')
    doc.add_paragraph()
    add_para(doc, f'生成时间：{datetime.now().strftime("%Y年%m月%d日")}',
             size=10, align='center')
    doc.add_page_break()

    # ── 1. 模型设定说明 ────────────────────────────────────────────────────
    add_heading(doc, '一、模型设定说明', level=1)

    add_para(doc,
             '本研究采用递进式普通最小二乘（OLS）回归方法，共设置五个嵌套模型（Model 0至Model 4），'
             '依次引入控制变量、标题文本特征、封面图像特征、双模态协同特征及交互效应，'
             '以系统检验各类特征对视频传播效果的影响。',
             indent=True, align='justify')
    add_para(doc,
             '为处理同一UP主发布多个视频所引起的组内相关问题，所有模型均采用以uid为聚类单元的'
             '聚类稳健标准误（Cluster-robust Standard Errors）。'
             '所有连续型核心解释变量均经过z-score标准化处理（均值=0，标准差=1），'
             '以便跨变量比较标准化回归系数大小。',
             indent=True, align='justify')

    add_heading(doc, '1.1 分类变量参照组设置', level=2)
    ref_table = doc.add_table(rows=1, cols=3)
    ref_table.style = 'Table Grid'
    ref_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = ref_table.rows[0].cells
    for cell, text in zip(hdr, ['分类变量', '变量说明', '参照组（基准类别）']):
        set_cell_style(cell, text, bold=True, bg_color='D6E4F0')
    categ_ref_data = [
        ('C(domain)',             '知识领域',    CATEG_REF.get('domain', '科学科普')),
        ('C(level)',              '用户等级',    f'等级 {CATEG_REF.get("level","1")}'),
        ('C(pub_month)',          '发布月份',    f'{CATEG_REF.get("pub_month","1")}月'),
        ('C(pub_weekday)',        '发布星期',    f'周{CATEG_REF.get("pub_weekday","一")}（星期1）'),
        ('C(pub_hour)',           '发布小时',    f'{CATEG_REF.get("pub_hour","0")}时'),
        ('C(complementarity_type)', '互补类型', CATEG_REF.get('complementarity_type','neutral')),
    ]
    for cn, desc, ref in categ_ref_data:
        row = ref_table.add_row().cells
        set_cell_style(row[0], cn)
        set_cell_style(row[1], desc)
        set_cell_style(row[2], ref)
    set_table_borders(ref_table)

    # ── 2. 主回归结果表 ────────────────────────────────────────────────────
    add_heading(doc, f'二、主回归结果（因变量：{VAR_LABELS.get(depvar,depvar)}）', level=1)

    model_names_avail = [m for m in model_order
                         if m in all_coef_dfs.get(depvar, {})]

    if not model_names_avail:
        add_para(doc, '无可用模型结果，请检查数据和变量设置。')
    else:
        # 表标题
        table_num = '1' if depvar == 'log_view' else '2'
        p_caption = doc.add_paragraph()
        run = p_caption.add_run(
            f'表{table_num}  递进式回归结果汇总（因变量：{VAR_LABELS.get(depvar,depvar)}）'
        )
        set_font(run, size=10, bold=True)
        p_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # 构建变量分区
        var_sections = [
            ('控制变量', CTRL_CONTINUOUS + CTRL_BINARY),
            ('控制变量（分类）', [f'C({c})[T.*]' for c in CTRL_CATEG]),
            ('标题特征变量', TITLE_VARS),
            ('封面特征变量', COVER_VARS),
            ('双模态协同变量', MODAL_VARS_BASE),
            ('双模态协同（分类）', [f'C(complementarity_type)[T.*]']),
            ('交互效应', INTERACTION_VARS),
        ]

        # 收集所有变量（按自然顺序）
        all_vars_ordered = []
        seen_v = set()
        for mn in model_names_avail:
            for v in all_coef_dfs[depvar][mn]['variable']:
                if v not in seen_v and v != 'Intercept':
                    all_vars_ordered.append(v)
                    seen_v.add(v)

        n_cols = 2 + len(model_names_avail)
        table = doc.add_table(rows=2, cols=n_cols)
        table.style = 'Table Grid'
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        # 表头行1：合并模型列
        hdr_row1 = table.rows[0].cells
        set_cell_style(hdr_row1[0], '变量', bold=True, bg_color='1F4E79', size=9)
        set_cell_style(hdr_row1[1], '中文别名', bold=True, bg_color='1F4E79', size=9)
        # 将文字颜色改为白色
        for ci in [0, 1]:
            for p in hdr_row1[ci].paragraphs:
                for run in p.runs:
                    run.font.color.rgb = RGBColor(255, 255, 255)
        for ci, mn in enumerate(model_names_avail):
            col_i = 2 + ci
            set_cell_style(hdr_row1[col_i],
                           f'{mn}\n{MODEL_NAMES_CN.get(mn,"")}',
                           bold=True, bg_color='2E75B6', size=9)
            for p in hdr_row1[col_i].paragraphs:
                for run in p.runs:
                    run.font.color.rgb = RGBColor(255, 255, 255)

        # 表头行2：系数格式说明
        hdr_row2 = table.rows[1].cells
        set_cell_style(hdr_row2[0], '变量名', bold=True, bg_color='D6E4F0', size=9)
        set_cell_style(hdr_row2[1], '中文别名', bold=True, bg_color='D6E4F0', size=9)
        for ci in range(len(model_names_avail)):
            set_cell_style(hdr_row2[2+ci], '系数\n(聚类稳健标准误)',
                           bold=True, bg_color='D6E4F0', size=8)

        # 数据行
        for ri, v in enumerate(all_vars_ordered):
            row = table.add_row().cells
            bg = 'F0F6FF' if ri % 2 == 1 else None
            is_categ_dummy = ('[T.' in v)
            display_name = v.replace('_std', '').replace('[T.', '=').replace(']', '')
            cn_label = VAR_LABELS.get(v, VAR_LABELS.get(v.replace('[T.', '').split(']')[0], v))

            set_cell_style(row[0], display_name, size=8, align='left', bg_color=bg)
            set_cell_style(row[1], cn_label, size=8, align='left', bg_color=bg)

            for ci, mn in enumerate(model_names_avail):
                cdf = all_coef_dfs[depvar][mn]
                match = cdf[cdf['variable'] == v]
                if not match.empty:
                    coef = match['coefficient'].values[0]
                    se   = match['std_err'].values[0]
                    sig  = match['significance'].values[0]
                    cell_text = f"{coef:.3f}{sig}\n({se:.3f})"
                else:
                    cell_text = '—'
                set_cell_style(row[2+ci], cell_text, size=8, bg_color=bg)

        # 截距行
        row_int = table.add_row().cells
        set_cell_style(row_int[0], 'Intercept', bold=True, size=8)
        set_cell_style(row_int[1], '截距', bold=True, size=8)
        for ci, mn in enumerate(model_names_avail):
            cdf = all_coef_dfs[depvar][mn]
            match = cdf[cdf['variable'] == 'Intercept']
            if not match.empty:
                coef = match['coefficient'].values[0]
                se   = match['std_err'].values[0]
                sig  = match['significance'].values[0]
                set_cell_style(row_int[2+ci], f"{coef:.3f}{sig}\n({se:.3f})",
                               bold=True, size=8)
            else:
                set_cell_style(row_int[2+ci], '—', size=8)

        # 模型统计区
        for stat_label, stat_key, fmt in [
            ('N（观测数）', 'N', '{:,}'),
            ('R²', 'R2', '{:.4f}'),
            ('调整R²', 'R2_adj', '{:.4f}'),
            ('ΔR²（vs Model 0）', 'delta_R2', '{:.4f}'),
            ('AIC', 'AIC', '{:.2f}'),
            ('BIC', 'BIC', '{:.2f}'),
        ]:
            row_s = table.add_row().cells
            r2_m0 = all_summaries.get(depvar,{}).get('Model 0',{}).get('R2', np.nan)
            set_cell_style(row_s[0], stat_key, bold=True, bg_color='EBF3FB', size=9)
            set_cell_style(row_s[1], stat_label, bold=True, bg_color='EBF3FB', size=9)
            for ci, mn in enumerate(model_names_avail):
                s = all_summaries.get(depvar,{}).get(mn, {})
                if stat_key == 'delta_R2':
                    r2_mn = s.get('R2', np.nan)
                    val = r2_mn - r2_m0 if not np.isnan(r2_mn) and not np.isnan(r2_m0) else np.nan
                else:
                    val = s.get(stat_key, np.nan)
                val_str = fmt.format(val) if (val is not None and not (isinstance(val, float) and np.isnan(val))) else '—'
                set_cell_style(row_s[2+ci], val_str, bg_color='EBF3FB',
                               bold=True, size=9)

        set_table_borders(table)

        # 表格注释
        p_note = doc.add_paragraph()
        run_note = p_note.add_run(
            '注：***p<0.001，**p<0.01，*p<0.05，†p<0.1；'
            '括号内为聚类稳健标准误（按UP主uid聚类）；'
            '所有连续变量已做z-score标准化（均值=0，标准差=1）；'
            f'分类变量参照组详见表注；'
            'ΔR²表示相对于Model 0的R²增量。'
        )
        set_font(run_note, size=8.5, italic=True)
        p_note.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # ── 3. 文字解释 ────────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, '三、回归结果文字解释', level=1)

    summaries_dv = all_summaries.get(depvar, {})
    coefs_dv     = all_coef_dfs.get(depvar, {})

    def get_r2(mn): return summaries_dv.get(mn, {}).get('R2', np.nan)
    def get_coef(mn, vname):
        cdf = coefs_dv.get(mn, pd.DataFrame())
        if cdf.empty: return None, None, None
        # 优先查找标准化版本
        match = cdf[cdf['variable'] == vname + '_std']
        if match.empty:
            match = cdf[cdf['variable'] == vname]
        if match.empty: return None, None, None
        return (match['coefficient'].values[0],
                match['p_value'].values[0],
                match['significance'].values[0])

    # 3.1 整体解释力
    add_heading(doc, '3.1 整体解释力提升分析', level=2)
    r2_vals = {mn: get_r2(mn) for mn in model_order
               if mn in summaries_dv}

    interp_lines = []
    if 'Model 0' in r2_vals and not np.isnan(r2_vals['Model 0']):
        interp_lines.append(
            f"控制变量模型（Model 0）的R²为{r2_vals['Model 0']:.4f}，"
            f"调整R²为{summaries_dv.get('Model 0',{}).get('R2_adj',float('nan')):.4f}，"
            f"说明UP主特征、视频时长等基础控制变量能解释因变量{r2_vals['Model 0']*100:.1f}%的方差。"
        )
    if 'Model 1' in r2_vals and 'Model 0' in r2_vals:
        delta = r2_vals['Model 1'] - r2_vals['Model 0']
        interp_lines.append(
            f"加入标题文本特征后（Model 1），R²提升至{r2_vals['Model 1']:.4f}，"
            f"较Model 0增加ΔR²={delta:.4f}（+{delta*100:.2f}个百分点），"
            f"{'说明标题特征对视频传播具有显著的增量解释力。' if delta > 0.005 else '标题特征的增量解释力相对有限。'}"
        )
    if 'Model 2' in r2_vals and 'Model 1' in r2_vals:
        delta = r2_vals['Model 2'] - r2_vals['Model 1']
        interp_lines.append(
            f"进一步加入封面图像特征（Model 2），R²提升至{r2_vals['Model 2']:.4f}，"
            f"较Model 1增加ΔR²={delta:.4f}，"
            f"{'封面特征显著提升了模型解释力。' if delta > 0.003 else '封面特征的独立解释力相对温和。'}"
        )
    if 'Model 3' in r2_vals and 'Model 2' in r2_vals:
        delta = r2_vals['Model 3'] - r2_vals['Model 2']
        interp_lines.append(
            f"双模态协同变量的引入（Model 3）使R²达到{r2_vals['Model 3']:.4f}，"
            f"ΔR²={delta:.4f}，"
            f"{'表明标题与封面的协同效应具有超越单一模态的额外解释价值，支持双模态框架的理论预设。' if delta > 0.002 else '双模态协同变量的增量解释力有限，但仍具有理论意义。'}"
        )
    if 'Model 4' in r2_vals and 'Model 3' in r2_vals:
        delta = r2_vals['Model 4'] - r2_vals['Model 3']
        interp_lines.append(
            f"加入交互效应（Model 4）后R²为{r2_vals['Model 4']:.4f}，"
            f"ΔR²={delta:.4f}，{'交互项进一步提升了模型拟合度。' if delta > 0.001 else '交互项对整体解释力的贡献较小。'}"
        )

    for line in interp_lines:
        add_para(doc, line, indent=True, align='justify')

    # 3.2 关键变量解释
    add_heading(doc, '3.2 关键解释变量分析', level=2)

    key_vars_explain = [
        ('title_sentiment_intensity', '标题情感强度',
         '情感强度越高的标题往往更能激发用户点击意愿，但过度情绪化可能引发用户质疑。'),
        ('curiosity_marker_count', '好奇心标记数',
         '好奇心标记（如"为什么""揭秘""你不知道的"）通过制造信息缺口激发用户点击。'),
        ('practical_value_word_count', '实用价值词频',
         '强调实用价值的标题（如"方法""技巧""攻略"）在知识类视频中受到用户偏好。'),
        ('image_brightness', '封面亮度',
         '适中的亮度水平有助于提高封面在信息流中的视觉辨识度。'),
        ('image_colorfulness', '封面色彩丰富度',
         '色彩丰富的封面在视觉上更具吸引力，但过度复杂可能分散注意力。'),
        ('face_presence', '封面人脸存在',
         '封面出现人脸利用了人类对面孔的天然注意偏向，可增强视频的点击吸引力。'),
        ('image_title_fit_centered', '标题-封面语义相似度',
         '标题与封面的语义一致性越高，用户认知负荷越低，传播效果可能越好。'),
        ('image_title_fit_sq_centered', '标题-封面语义相似度（平方项）',
         '若平方项显著，说明语义相似度对传播效果存在非线性（如倒U形）关系，即适度匹配优于过度相似或完全不相似。'),
        ('sentiment_consistency', '情感一致性',
         '标题与封面情感方向一致可减少认知失调，提升用户信任。'),
        ('sentiment_conflict', '情感冲突',
         '适度的情感对比（如严肃封面+煽情标题）可能制造悬念感，但过度冲突会降低可信度。'),
        ('semantic_text_interaction', '语义文本交互项',
         '该交互项检验语义相似度与标题文本丰富度的联合效应，揭示双模态协同的细粒度机制。'),
    ]

    for vname, cn_name, interp_template in key_vars_explain:
        coef, pval, sig = get_coef('Model 3', vname)
        if coef is None:
            coef, pval, sig = get_coef('Model 4', vname)
        if coef is None:
            continue
        direction = '正向' if coef > 0 else '负向'
        sig_word  = ('在0.1%水平上显著' if sig == '***' else
                     '在1%水平上显著' if sig == '**' else
                     '在5%水平上显著' if sig == '*' else
                     '在10%水平上边际显著' if sig == '†' else '不显著')
        line = (f"【{cn_name}（{vname}）】：标准化系数β={coef:.3f}，{sig_word}（p={pval:.4f}）。"
                f"{interp_template}")
        add_para(doc, line, indent=True, align='justify')

    # 3.3 非线性关系检验
    add_heading(doc, '3.3 image_title_fit 非线性关系检验', level=2)
    coef_lin, pval_lin, sig_lin = get_coef('Model 3', 'image_title_fit_centered')
    coef_sq,  pval_sq,  sig_sq  = get_coef('Model 3', 'image_title_fit_sq_centered')

    if coef_lin is not None and coef_sq is not None:
        nl_text = (
            f"在双模态协同模型（Model 3）中，标题-封面语义相似度线性项的标准化系数为β={coef_lin:.3f}"
            f"（{sig_lin}，p={pval_lin:.4f}），平方项系数为β²={coef_sq:.3f}"
            f"（{sig_sq}，p={pval_sq:.4f}）。"
        )
        if sig_sq in ['*', '**', '***', '†']:
            if coef_sq < 0:
                nl_text += ("平方项系数显著为负，表明标题-封面语义相似度对视频传播的影响呈倒U形关系，"
                            "即中等程度的语义契合度（而非完全相似）最有利于传播效果，"
                            "与'适度新奇性'理论（Moderate Novelty Theory）相吻合。")
            else:
                nl_text += ("平方项系数显著为正，表明语义相似度对传播效果的影响呈U形关系，"
                            "极高或极低的一致性均比中等水平更有利，暗示内容的一致性信号可能存在门槛效应。")
        else:
            nl_text += ("平方项系数不显著（p>{:.3f}），不支持语义相似度与传播效果之间的非线性关系假设，"
                        "现有数据更符合线性影响模式。".format(pval_sq))
        add_para(doc, nl_text, indent=True, align='justify')
    else:
        add_para(doc, 'image_title_fit 相关变量在当前数据中不可用，无法进行非线性检验。',
                 indent=True)

    # 3.4 交互效应解释
    add_heading(doc, '3.4 交互效应（Model 4）解释', level=2)
    interact_text = (
        "Model 4在Model 3基础上引入了三组交互项：（1）semantic_text_interaction代表语义相似度与"
        "标题文本质量的乘积交互，若系数显著为正，说明高语义协同度与高质量标题形成互补增益效应，"
        "即好封面+好标题的组合效果超越各自的单独效果之和；（2）image_title_fit × C(domain)"
        "交互项检验语义协同效应是否因知识领域而异，若显著，说明某些领域（如财经商业）的用户"
        "对视觉-文字一致性更为敏感；（3）sentiment_consistency × C(domain)检验情感一致性效应"
        "是否具有领域异质性，例如科学科普类视频用户或更重视信息确定性而非情感激发。"
    )
    add_para(doc, interact_text, indent=True, align='justify')

    # ── 4. 模型诊断 ────────────────────────────────────────────────────────
    add_heading(doc, '四、模型诊断', level=1)

    add_heading(doc, '4.1 多重共线性检验（VIF）', level=2)
    add_para(doc,
             '建议对Model 3中的核心解释变量进行VIF检验。VIF > 10通常视为严重共线性，'
             'VIF > 5为中等共线性。本研究对连续变量进行了z-score标准化，有助于缓解由量纲差异引起的数值共线性。'
             '若image_title_fit_centered与image_title_fit_sq_centered之间存在高相关，'
             '可考虑对原始变量中心化后再构造平方项（本研究已采用中心化设计）。',
             indent=True, align='justify')

    add_heading(doc, '4.2 残差正态性', level=2)
    add_para(doc,
             '由于样本量较大（N>' +
             str(min([s.get('N', 0) for s in summaries_dv.values() if s], default=0)) +
             '），根据中心极限定理，OLS估计量的推断结果对残差轻度偏离正态仍具有稳健性。'
             '使用聚类稳健标准误进一步保证了大样本下的统计推断可靠性。',
             indent=True, align='justify')

    add_heading(doc, '4.3 异方差处理', level=2)
    add_para(doc,
             '本研究已采用聚类稳健标准误（Cluster-Robust SE）替代传统OLS标准误，'
             '该方法在大样本条件下对组内任意形式的异方差和自相关均具有一致性。',
             indent=True, align='justify')

    # ── 保存 ─────────────────────────────────────────────────────────────────
    doc.save(filepath)
    logger.info(f"Word 报告已保存：{filepath}")

# ─────────────────────────────────────────────────────────────────────────────
# 14. 系数图绘制
# ─────────────────────────────────────────────────────────────────────────────
# 核心解释变量（用于系数图，排除控制变量哑变量）
CORE_VARS_FOR_PLOT = (
    TITLE_VARS + COVER_VARS + MODAL_VARS_BASE + INTERACTION_VARS
)

def plot_coefficient(depvar, model_name, coef_df, save_path, core_vars):
    """
    绘制单个模型的核心变量系数图（含置信区间）
    """
    if coef_df is None or coef_df.empty:
        logger.warning(f"系数图：{model_name} 无数据，跳过")
        return

    # 筛选核心变量（标准化版本）
    core_vars_std = [v + '_std' for v in core_vars] + core_vars
    plot_df = coef_df[
        coef_df['variable'].isin(core_vars_std) &
        (coef_df['variable'] != 'Intercept')
    ].copy()

    if plot_df.empty:
        logger.warning(f"系数图：{model_name} 无核心变量可绘制，跳过")
        return

    # 清理变量名（去除 _std 后缀）
    plot_df['var_display'] = (
        plot_df['variable']
        .str.replace('_std', '', regex=False)
        .map(lambda x: VAR_LABELS.get(x, x))
    )

    # 按系数绝对值排序
    plot_df = plot_df.sort_values('coefficient', key=abs, ascending=True)

    # 颜色：正系数蓝色，负系数橙色
    colors = ['#2196F3' if c > 0 else '#FF7043' for c in plot_df['coefficient']]
    sig_alpha = [1.0 if p < 0.05 else 0.45 for p in plot_df['p_value']]

    fig_height = max(4, len(plot_df) * 0.42 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = range(len(plot_df))
    bars = ax.barh(list(y_pos),
            plot_df['coefficient'].values,
            xerr=[plot_df['coefficient'].values - plot_df['ci_lower'].values,
                  plot_df['ci_upper'].values - plot_df['coefficient'].values],
            color=colors,
            edgecolor='white',
            linewidth=0.5,
            height=0.65,
            error_kw={'elinewidth': 1.2, 'ecolor': '#333333', 'capsize': 3})
    # 新版 matplotlib 不再支持向 bar 传入 alpha 列表，逐根设置透明度
    for bar, bar_alpha in zip(bars, sig_alpha):
        bar.set_alpha(bar_alpha)

    # 添加系数标签
    for i, (coef, pval, sig) in enumerate(zip(
            plot_df['coefficient'].values,
            plot_df['p_value'].values,
            plot_df['significance'].values)):
        ax.text(coef + (0.005 if coef >= 0 else -0.005),
                i, f'{coef:.3f}{sig}',
                va='center', ha='left' if coef >= 0 else 'right',
                fontsize=7.5, color='#222222')

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df['var_display'].values, fontsize=8)
    ax.axvline(0, color='#333333', linewidth=1.0, linestyle='-')
    ax.set_xlabel('标准化回归系数 (β)', fontsize=10)

    cn_depvar = VAR_LABELS.get(depvar, depvar)
    ax.set_title(f'{MODEL_NAMES_CN.get(model_name, model_name)}\n'
                 f'因变量：{cn_depvar} | 聚类稳健标准误（uid）',
                 fontsize=11, pad=12)

    # 显著性图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', alpha=1.0, label='正向显著(p<0.05)'),
        Patch(facecolor='#FF7043', alpha=1.0, label='负向显著(p<0.05)'),
        Patch(facecolor='#AAAAAA', alpha=0.45, label='不显著(p≥0.05)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.8)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"系数图已保存：{save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 15. 执行输出
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ORDER = ['Model 0', 'Model 1', 'Model 2', 'Model 3', 'Model 4']

# 15.1 Excel
logger.info("\n" + "="*50)
logger.info("输出 Excel 回归结果表...")
write_excel_results(
    os.path.join(OUT_DIR, "regression_results.xlsx"),
    all_coef_dfs, all_summaries, all_results, MODEL_ORDER
)

# 15.2 Word 报告
logger.info("\n输出 Word 回归报告...")
for depvar in DEPVARS:
    if depvar not in all_results or not all_results[depvar]:
        logger.warning(f"{depvar} 无模型结果，跳过 Word 输出")
        continue
    fname = f"regression_results_{depvar}.docx"
    write_word_report(
        os.path.join(OUT_DIR, fname),
        depvar, all_coef_dfs, all_summaries, all_results, MODEL_ORDER, formula_log
    )

# 15.3 系数图（每个模型 × 每个因变量）
logger.info("\n输出系数图...")
for depvar in DEPVARS:
    for model_name in MODEL_ORDER:
        coef_df = all_coef_dfs.get(depvar, {}).get(model_name)
        if coef_df is not None:
            fig_name = (f"coefficient_plot_core_variables"
                        f"_{depvar}_{model_name.replace(' ','')}.png")
            plot_coefficient(
                depvar, model_name, coef_df,
                os.path.join(FIG_DIR, fig_name),
                CORE_VARS_FOR_PLOT
            )

# ─────────────────────────────────────────────────────────────────────────────
# 16. 生成 Markdown 分析报告
# ─────────────────────────────────────────────────────────────────────────────
logger.info("\n生成 Markdown 回归分析报告...")

md_lines = []
md_lines.append("# 第八步：递进式回归建模报告")
md_lines.append(f"\n**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
md_lines.append(f"\n**数据文件：** `{INPUT_FILE}`")
md_lines.append(f"\n**建模样本量：** {len(df_model):,} 个视频")
md_lines.append(f"\n**聚类单元：** uid（UP主），共 {df_model['uid'].nunique():,} 个聚类")

md_lines.append("\n---\n")
md_lines.append("## 一、变量设置")
md_lines.append(f"\n- **因变量：** {', '.join(DEPVARS)}")
md_lines.append(f"- **控制变量（连续）：** {', '.join(CTRL_CONTINUOUS)}")
md_lines.append(f"- **控制变量（分类）：** {', '.join(CTRL_CATEG)}")
md_lines.append(f"- **标题特征：** {', '.join(TITLE_VARS)}")
md_lines.append(f"- **封面特征：** {', '.join(COVER_VARS)}")
md_lines.append(f"- **双模态协同：** {', '.join(MODAL_VARS_BASE + MODAL_CATEG)}")
md_lines.append(f"- **交互效应：** {', '.join(INTERACTION_VARS)}")

md_lines.append("\n### 分类变量参照组\n")
md_lines.append("| 变量 | 参照组 |")
md_lines.append("|------|--------|")
for k, v in CATEG_REF.items():
    md_lines.append(f"| C({k}) | {v} |")

md_lines.append("\n---\n")
md_lines.append("## 二、模型拟合统计")

for depvar in DEPVARS:
    md_lines.append(f"\n### 因变量：{VAR_LABELS.get(depvar, depvar)} (`{depvar}`)\n")
    md_lines.append("| 模型 | N | R² | 调整R² | ΔR²(vs M0) | AIC | BIC |")
    md_lines.append("|------|---|-----|--------|------------|-----|-----|")
    r2_m0 = all_summaries.get(depvar, {}).get('Model 0', {}).get('R2', np.nan)
    for mn in MODEL_ORDER:
        s = all_summaries.get(depvar, {}).get(mn)
        if not s:
            md_lines.append(f"| {mn} | — | — | — | — | — | — |")
            continue
        delta = s['R2'] - r2_m0 if not np.isnan(r2_m0) else np.nan
        md_lines.append(
            f"| {mn} | {s['N']:,} | {s['R2']:.4f} | {s['R2_adj']:.4f} | "
            f"{'—' if np.isnan(delta) else f'+{delta:.4f}'} | "
            f"{s['AIC']:.1f} | {s['BIC']:.1f} |"
        )

md_lines.append("\n---\n")
md_lines.append("## 三、缺失变量报告\n")
if missing_report:
    for group, vars_miss in missing_report.items():
        md_lines.append(f"- **{group}**：缺失 {vars_miss}")
else:
    md_lines.append("[OK] 所有预设变量均在数据中找到，无缺失。")

md_lines.append("\n---\n")
md_lines.append("## 四、实际使用公式\n")
for depvar in DEPVARS:
    md_lines.append(f"\n### 因变量：{depvar}\n")
    for mn in MODEL_ORDER:
        f = formula_log.get(depvar, {}).get(mn, '（未运行）')
        md_lines.append(f"**{mn}：**\n```\n{f}\n```\n")

md_lines.append("\n---\n")
md_lines.append("## 五、输出文件清单\n")
output_files = [
    ("outputs/regression_results.xlsx", "Excel格式完整回归结果（多Sheet）"),
    ("outputs/regression_results_log_view.docx", "Word回归报告（log_view）"),
    ("outputs/regression_results_log_view_per_day.docx", "Word回归报告（log_view_per_day）"),
    ("outputs/logs/regression_model_report.md", "本分析报告（Markdown）"),
]
for fn, desc in output_files:
    md_lines.append(f"- `{fn}` — {desc}")
md_lines.append("\n**系数图（每模型×每因变量）：**")
for depvar in DEPVARS:
    for mn in MODEL_ORDER:
        fn = (f"outputs/figures/coefficient_plot_core_variables"
              f"_{depvar}_{mn.replace(' ','')}.png")
        md_lines.append(f"- `{fn}`")

md_lines.append("\n---\n")
md_lines.append("## 六、方法说明\n")
md_lines.append(
    "1. **回归方法：** OLS（普通最小二乘法）+ 聚类稳健标准误（按uid）\n"
    "2. **变量标准化：** 所有连续核心解释变量做z-score标准化（均值0，标准差1）\n"
    "3. **分类变量：** 自动构造哑变量，参照组见上表\n"
    "4. **因变量：** log_view（对数总播放量），log_view_per_day（对数日均播放量）\n"
    "5. **缺失值处理：** 删除核心变量缺失的观测，已记录缺失行数\n"
    "6. **显著性标注：** *** p<0.001，** p<0.01，* p<0.05，† p<0.1\n"
    "7. **软件环境：** Python + statsmodels（OLS + cluster covariance）\n"
)

md_lines.append("\n---\n")
md_lines.append("*本报告由 step8_regression_modeling.py 自动生成。*")

with open(log_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

logger.info(f"Markdown报告已保存：{log_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 17. 完成汇总
# ─────────────────────────────────────────────────────────────────────────────
logger.info("\n" + "="*60)
logger.info("第八步：递进式回归建模 完成")
logger.info("="*60)
logger.info("输出文件清单：")
logger.info(f"  [Excel]  outputs/regression_results.xlsx")
for depvar in DEPVARS:
    logger.info(f"  [Word]   outputs/regression_results_{depvar}.docx")
logger.info(f"  [MD]     outputs/logs/regression_model_report.md")
logger.info(f"  [Log]    outputs/logs/regression_run.log")
for depvar in DEPVARS:
    for mn in MODEL_ORDER:
        fn = f"outputs/figures/coefficient_plot_core_variables_{depvar}_{mn.replace(' ','')}.png"
        logger.info(f"  [Figure] {fn}")
logger.info("="*60)
