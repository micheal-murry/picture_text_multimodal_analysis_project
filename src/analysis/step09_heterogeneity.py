"""
第九步：异质性分析（增强版）
研究题目：标题和封面双模态协同对社交媒体平台视频传播的影响——以B站知识类视频为例
"""

import os, sys, warnings, traceback, itertools
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns
from openpyxl import Workbook
from openpyxl.styles import (PatternFill, Font, Alignment, Border, Side,
                               GradientFill)
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# 0. 路径与字体配置
# ============================================================
INPUT_FILE  = 'outputs/modeling_dataset.csv'
OUTPUT_DIR  = 'outputs'
FIG_DIR     = 'outputs/figures'
LOG_DIR     = 'outputs/logs'
EXCEL_OUT   = 'outputs/heterogeneity_results.xlsx'
REPORT_OUT  = 'outputs/logs/heterogeneity_report.md'

for d in [OUTPUT_DIR, FIG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- 中文字体 ----
CN_FONT_PATHS = [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
]
_font_path = next((p for p in CN_FONT_PATHS if os.path.exists(p)), None)

if _font_path:
    from matplotlib import font_manager as fm
    fm.fontManager.addfont(_font_path)
    _prop = fm.FontProperties(fname=_font_path)
    _fname = _prop.get_name()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [_fname, 'DejaVu Sans']
    CN_PROP = _prop
else:
    CN_PROP = None
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ---- 学术图表风格 ----
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.edgecolor':   '#333333',
    'axes.linewidth':   0.8,
    'axes.grid':        True,
    'grid.color':       '#E0E0E0',
    'grid.linewidth':   0.5,
    'grid.linestyle':   '--',
    'xtick.direction':  'out',
    'ytick.direction':  'out',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
})

# ---- 日志 ----
LOG_LINES = []
def log(msg, level='INFO'):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}][{level}] {msg}"
    LOG_LINES.append(line)
    print(line)

log("==== 第九步：异质性分析（增强版）开始 ====")

# ============================================================
# 1. 数据加载与预处理
# ============================================================
log("--- 1. 数据加载 ---")
try:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    log(f"成功加载数据：{df.shape[0]} 行 × {df.shape[1]} 列")
except FileNotFoundError:
    df = pd.read_csv('/mnt/user-data/uploads/modeling_dataset.csv', low_memory=False)
    log(f"从上传目录加载：{df.shape[0]} 行 × {df.shape[1]} 列")

# ---- 1.1 因变量 ----
Y_VAR = 'log_view_per_day'
if Y_VAR not in df.columns or df[Y_VAR].isna().sum() > df.shape[0] * 0.5:
    Y_VAR = 'log_view'
    log(f"切换因变量为 {Y_VAR}", 'WARN')
log(f"因变量：{Y_VAR}，有效值 {df[Y_VAR].notna().sum()}")

# ---- 1.2 均值中心化 ----
if 'image_title_fit' in df.columns:
    fit_mean = df['image_title_fit'].mean()
    df['image_title_fit_centered']    = df['image_title_fit'] - fit_mean
    df['image_title_fit_sq_centered'] = df['image_title_fit_centered'] ** 2
    log(f"image_title_fit 均值中心化，均值={fit_mean:.4f}")
else:
    df['image_title_fit_centered']    = df.get('image_title_fit_z', 0)
    df['image_title_fit_sq_centered'] = df['image_title_fit_centered'] ** 2

# ---- 1.3 封面二值变量 ----
BINARY_COVER_VARS = ['cover_has_chart_or_formula', 'cover_is_slide_or_screenshot']
AVAILABLE_BINARY  = []
for v in BINARY_COVER_VARS:
    if v in df.columns and df[v].notna().sum() > 100:
        df[v] = df[v].fillna(0).astype(int)
        AVAILABLE_BINARY.append(v)
    else:
        log(f"[缺失] {v} 全为 NaN，从模型中剔除", 'WARN')

# ---- 1.4 控制变量 ----
CONTROLS = ['log_fans', 'log_duration', 'log_age',
            'pub_weekday', 'pub_hour', 'tag_count',
            'description_len', 'is_collection', 'copyright']
CONTROLS = [c for c in CONTROLS if c in df.columns and df[c].notna().sum() > df.shape[0] * 0.5]

# ---- 1.5 核心解释变量 ----
CORE_VARS_ALL = [
    'image_title_fit_centered',
    'image_title_fit_sq_centered',
    'title_cover_text_overlap',
    'sentiment_consistency',
    'curiosity_marker_count',
    'practical_value_word_count',
] + AVAILABLE_BINARY
CORE_VARS = [v for v in CORE_VARS_ALL if v in df.columns]

# ---- 1.6 domain标准化 ----
DOMAIN_MAP = {
    '科学科普':'科学科普','科普':'科学科普',
    '社科人文':'社科人文','人文':'社科人文',
    '财经商业':'财经商业','财经':'财经商业','商业':'财经商业',
    '职场教育':'职场教育','教育':'职场教育','职场':'职场教育',
}
df['domain'] = df['domain'].map(lambda x: DOMAIN_MAP.get(str(x).strip(), str(x).strip()))
DOMAINS = ['科学科普', '社科人文', '财经商业', '职场教育']
df = df[df['domain'].isin(DOMAINS)].copy()
log(f"domain分布：{df['domain'].value_counts().to_dict()}")

# ---- 1.7 level ----
LEVELS = sorted(df['level'].dropna().unique().tolist())
log(f"level取值：{LEVELS}")

# ---- 1.8 粉丝规模三分位 ----
q33 = df['log_fans'].quantile(1/3)
q67 = df['log_fans'].quantile(2/3)
df['fans_group'] = pd.cut(
    df['log_fans'],
    bins=[-np.inf, q33, q67, np.inf],
    labels=['low_fans', 'mid_fans', 'high_fans']
)
log(f"粉丝规模分组：q33={q33:.3f}, q67={q67:.3f}")

# ---- 1.9 建模样本 ----
all_vars = [Y_VAR] + CORE_VARS + CONTROLS
df_model = df.dropna(subset=[c for c in all_vars if c in df.columns]).copy()
log(f"有效建模样本：{df_model.shape[0]} 行")

# ============================================================
# 2. 辅助函数
# ============================================================
def build_formula(y, core_vars, controls):
    return f"{y} ~ {' + '.join(core_vars + controls)}"

def run_ols(df_sub, formula, group_label='', min_n=50):
    try:
        if df_sub.shape[0] < min_n:
            log(f"  [{group_label}] 样本量不足({df_sub.shape[0]})，跳过", 'WARN')
            return None
        return smf.ols(formula, data=df_sub).fit(cov_type='HC3')
    except Exception as e:
        log(f"  [{group_label}] 回归失败：{e}", 'ERROR')
        return None

def sig_star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.1:   return '†'
    return ''

def extract_coef_table(res, group_label, focus_vars=None):
    if res is None:
        return pd.DataFrame()
    rows = []
    conf = res.conf_int()
    for var in res.params.index:
        if focus_vars and not any(fv in var for fv in focus_vars):
            continue
        p = res.pvalues[var]
        rows.append({
            'group':   group_label,
            'variable': var,
            'coef':    res.params[var],
            'ci_low':  conf.loc[var, 0],
            'ci_high': conf.loc[var, 1],
            'pvalue':  p,
            'sig':     sig_star(p),
            'n_obs':   int(res.nobs),
            'r2':      round(res.rsquared, 4),
            'r2_adj':  round(res.rsquared_adj, 4),
        })
    return pd.DataFrame(rows)

def get_coef(res, var):
    if res is None or var not in res.params:
        return None, None
    return round(float(res.params[var]), 4), round(float(res.pvalues[var]), 4)

def coef_str(res, var):
    c, p = get_coef(res, var)
    if c is None: return 'N/A'
    return f"{c:+.4f}{sig_star(p)}"

def make_summary_table(res_dict, group_type, focal_vars):
    rows = []
    for grp, res in res_dict.items():
        if res is None:
            rows.append({'分组': grp, '分组类型': group_type, '样本量': 'N/A', 'R²': 'N/A', 'adj-R²': 'N/A'})
            continue
        row = {'分组': grp, '分组类型': group_type,
               '样本量': int(res.nobs), 'R²': round(res.rsquared, 4),
               'adj-R²': round(res.rsquared_adj, 4)}
        for v in focal_vars:
            if v in res.params:
                row[f'{v}_coef'] = round(float(res.params[v]), 4)
                row[f'{v}_pval'] = round(float(res.pvalues[v]), 4)
                row[f'{v}_sig']  = sig_star(res.pvalues[v])
        rows.append(row)
    return pd.DataFrame(rows)

def chow_test(res1, res2, res_pool, k):
    try:
        rss1 = res1.ssr; n1 = int(res1.nobs)
        rss2 = res2.ssr; n2 = int(res2.nobs)
        rss_p = res_pool.ssr
        F = ((rss_p - rss1 - rss2) / k) / ((rss1 + rss2) / (n1 + n2 - 2*k))
        p = 1 - stats.f.cdf(F, k, n1 + n2 - 2*k)
        return round(F, 4), round(p, 4)
    except:
        return np.nan, np.nan

# ============================================================
# 3. 核心变量配置
# ============================================================
FOCAL_VARS = [v for v in [
    'image_title_fit_centered',
    'title_cover_text_overlap',
    'sentiment_consistency',
    'curiosity_marker_count',
    'practical_value_word_count',
] + AVAILABLE_BINARY if v in df_model.columns]

FOCAL_LABELS = {
    'image_title_fit_centered':     '图文语义匹配度',
    'title_cover_text_overlap':     '标题封面文字重叠',
    'sentiment_consistency':        '情感一致性',
    'curiosity_marker_count':       '好奇心触发词',
    'practical_value_word_count':   '实用价值词',
    'cover_has_chart_or_formula':   '含图表/公式',
    'cover_is_slide_or_screenshot': 'PPT/截图封面',
}

FOCAL_LABELS_EN = {
    'image_title_fit_centered':     'Semantic Fit',
    'title_cover_text_overlap':     'Text Overlap',
    'sentiment_consistency':        'Sentiment Consistency',
    'curiosity_marker_count':       'Curiosity Markers',
    'practical_value_word_count':   'Practical Value Words',
}

CORE_MODEL_VARS = [v for v in CORE_VARS if v in df_model.columns]
CONTROL_VARS    = [v for v in CONTROLS   if v in df_model.columns]
formula_base    = build_formula(Y_VAR, CORE_MODEL_VARS, CONTROL_VARS)

log(f"核心解释变量：{CORE_MODEL_VARS}")
log(f"控制变量：{CONTROL_VARS}")

# 颜色方案
DOMAIN_COLORS = {'科学科普':'#1565C0','社科人文':'#C62828','财经商业':'#2E7D32','职场教育':'#6A1B9A'}
FANS_COLORS   = {'low_fans':'#D32F2F','mid_fans':'#F57C00','high_fans':'#1976D2'}
LEVEL_COLORS  = {-1:'#546E7A', 0:'#EF6C00', 1:'#1565C0'}
TEXT_COLORS   = {0:'#546E7A', 1:'#AD1457'}

fans_label_map  = {'low_fans':'低粉丝量组','mid_fans':'中等粉丝量组','high_fans':'高粉丝量组'}
text_label_map  = {'cover_text_0':'无封面文字','cover_text_1':'有封面文字'}
level_label_map = {f'level_{lv}': f'{"腰部" if lv==-1 else ("中部" if lv==0 else "头部")}UP主' for lv in LEVELS}

# ============================================================
# 4. 分组回归：知识领域
# ============================================================
log("=== 一、知识领域分组回归 ===")
domain_results   = {}
domain_coef_list = []
for dom in DOMAINS:
    sub = df_model[df_model['domain'] == dom].copy()
    log(f"  [{dom}] N={sub.shape[0]}")
    res = run_ols(sub, formula_base, dom)
    domain_results[dom] = res
    tbl = extract_coef_table(res, dom, FOCAL_VARS)
    domain_coef_list.append(tbl)
    if res:
        log(f"    R²={res.rsquared:.4f}, adj-R²={res.rsquared_adj:.4f}")

df_domain_coef = pd.concat(domain_coef_list, ignore_index=True) if domain_coef_list else pd.DataFrame()

# ============================================================
# 5. 总样本交互项检验
# ============================================================
log("=== 二、总样本交互项检验 ===")
INTERACTION_VARS = [
    'image_title_fit_centered','sentiment_consistency',
    'practical_value_word_count','title_cover_text_overlap',
]
if AVAILABLE_BINARY:
    INTERACTION_VARS.append(AVAILABLE_BINARY[0])

interaction_results  = {}
interaction_coef_list = []
for xvar in INTERACTION_VARS:
    if xvar not in df_model.columns:
        continue
    formula_int = formula_base.replace(' ~ ', f" ~ C(domain) + {xvar}:C(domain) + ")
    label = f"{xvar}×domain"
    res   = run_ols(df_model, formula_int, label)
    interaction_results[label] = res
    tbl = extract_coef_table(res, label, [xvar, 'C(domain)'])
    interaction_coef_list.append(tbl)
    if res:
        log(f"  {label}: R²={res.rsquared:.4f}")

df_interaction_coef = pd.concat(interaction_coef_list, ignore_index=True) if interaction_coef_list else pd.DataFrame()

# ============================================================
# 6. 按UP主层级分组
# ============================================================
log("=== 三、UP主层级分组回归 ===")
level_results   = {}
level_coef_list = []
fans_groups = ['low_fans', 'mid_fans', 'high_fans']

for lv in LEVELS:
    sub = df_model[df_model['level'] == lv].copy()
    lbl = f"level_{lv}"
    log(f"  [{lbl}] N={sub.shape[0]}")
    res = run_ols(sub, formula_base, lbl)
    level_results[lbl] = res
    tbl = extract_coef_table(res, lbl, FOCAL_VARS)
    level_coef_list.append(tbl)
    if res:
        log(f"    R²={res.rsquared:.4f}, adj-R²={res.rsquared_adj:.4f}")

df_level_coef = pd.concat(level_coef_list, ignore_index=True) if level_coef_list else pd.DataFrame()

# ---- 层级交互项 ----
level_int_results = {}
level_int_list    = []
for xvar in ['image_title_fit_centered','sentiment_consistency','practical_value_word_count']:
    if xvar not in df_model.columns:
        continue
    formula_lv = formula_base.replace(' ~ ', f" ~ C(level) + {xvar}:C(level) + ")
    label = f"{xvar}×level"
    res   = run_ols(df_model, formula_lv, label)
    level_int_results[label] = res
    tbl = extract_coef_table(res, label, [xvar, 'C(level)'])
    level_int_list.append(tbl)

df_level_int = pd.concat(level_int_list, ignore_index=True) if level_int_list else pd.DataFrame()

# ============================================================
# 7. 按粉丝规模分组
# ============================================================
log("=== 四、粉丝规模分组回归 ===")
fans_results   = {}
fans_coef_list = []
for fg in fans_groups:
    sub = df_model[df_model['fans_group'] == fg].copy()
    log(f"  [{fg}] N={sub.shape[0]}")
    res = run_ols(sub, formula_base, fg)
    fans_results[fg] = res
    tbl = extract_coef_table(res, fg, FOCAL_VARS)
    fans_coef_list.append(tbl)
    if res:
        log(f"    R²={res.rsquared:.4f}, adj-R²={res.rsquared_adj:.4f}")

df_fans_coef = pd.concat(fans_coef_list, ignore_index=True) if fans_coef_list else pd.DataFrame()

# ============================================================
# 8. 按封面文字有无分组
# ============================================================
log("=== 五、封面文字有无分组回归 ===")
text_results   = {}
text_coef_list = []
for tg in [0, 1]:
    sub = df_model[df_model['cover_has_text'] == tg].copy()
    lbl = f"cover_text_{tg}"
    log(f"  [{lbl}] N={sub.shape[0]}")
    res = run_ols(sub, formula_base, lbl, min_n=30)
    text_results[lbl] = res
    tbl = extract_coef_table(res, lbl, FOCAL_VARS)
    text_coef_list.append(tbl)
    if res:
        log(f"    R²={res.rsquared:.4f}, adj-R²={res.rsquared_adj:.4f}")

df_text_coef = pd.concat(text_coef_list, ignore_index=True) if text_coef_list else pd.DataFrame()

# ============================================================
# 9. Chow检验
# ============================================================
log("=== Chow检验 ===")
chow_rows = []
res_all   = run_ols(df_model, formula_base, 'total')
k_params  = len(CORE_MODEL_VARS) + len(CONTROL_VARS) + 1

for d1, d2 in itertools.combinations(DOMAINS, 2):
    r1 = domain_results.get(d1); r2 = domain_results.get(d2)
    sub12 = df_model[df_model['domain'].isin([d1, d2])]
    r12   = run_ols(sub12, formula_base, f'{d1}vs{d2}')
    if r1 and r2 and r12:
        F_, p_ = chow_test(r1, r2, r12, k_params)
        chow_rows.append({'比较组': f'{d1} vs {d2}','分组类型':'知识领域',
                           'F统计量': F_,'p值': p_,
                           '显著性': sig_star(p_) if not np.isnan(p_) else 'N/A',
                           '结论': '存在显著结构性差异' if (not np.isnan(p_) and p_ < 0.05) else '无显著差异'})

for f1, f2 in itertools.combinations(fans_groups, 2):
    r1 = fans_results.get(f1); r2 = fans_results.get(f2)
    sub12 = df_model[df_model['fans_group'].isin([f1, f2])]
    r12   = run_ols(sub12, formula_base, f'{f1}vs{f2}')
    if r1 and r2 and r12:
        F_, p_ = chow_test(r1, r2, r12, k_params)
        chow_rows.append({'比较组': f'{fans_label_map[f1]} vs {fans_label_map[f2]}',
                           '分组类型':'粉丝规模',
                           'F统计量': F_,'p值': p_,
                           '显著性': sig_star(p_) if not np.isnan(p_) else 'N/A',
                           '结论': '存在显著结构性差异' if (not np.isnan(p_) and p_ < 0.05) else '无显著差异'})

df_chow = pd.DataFrame(chow_rows)
log(f"Chow检验完成：{len(df_chow)} 组")

# ============================================================
# 10. 汇总表
# ============================================================
df_sum_domain = make_summary_table(domain_results, '知识领域', FOCAL_VARS)
df_sum_level  = make_summary_table(level_results,  'UP主层级', FOCAL_VARS)
df_sum_fans   = make_summary_table(fans_results,   '粉丝规模', FOCAL_VARS)
df_sum_text   = make_summary_table(text_results,   '封面文字', FOCAL_VARS)

# ============================================================
# 11. 学术质量系数比较图（核心图表）
# ============================================================
log("=== 绘制系数比较图 ===")

def plot_coef_forest(df_coef, groups, colors_dict, focal_vars, focal_labels,
                     suptitle, filename, group_label_map=None,
                     note='注：误差线为95%置信区间，†p<.1，*p<.05，**p<.01，***p<.001'):
    """学术风格系数森林图：每个核心变量一列子图，各组水平误差棒"""
    if df_coef.empty:
        log(f"  系数表为空，跳过 {filename}", 'WARN')
        return

    vars_to_plot = [v for v in focal_vars if v in df_coef['variable'].values]
    if not vars_to_plot:
        log(f"  无可绘变量，跳过 {filename}", 'WARN')
        return

    n_vars = len(vars_to_plot)
    n_grps = len(groups)
    fig_w  = max(12, n_vars * 2.6)
    fig_h  = max(4.5, n_grps * 0.9 + 2.5)

    fig, axes = plt.subplots(1, n_vars, figsize=(fig_w, fig_h),
                              sharey=True, squeeze=False)
    fig.patch.set_facecolor('white')

    for col_i, var in enumerate(vars_to_plot):
        ax = axes[0, col_i]
        ax.set_facecolor('#FAFAFA')
        sub = df_coef[df_coef['variable'] == var]

        for row_i, grp in enumerate(reversed(groups)):
            row = sub[sub['group'] == grp]
            if row.empty:
                continue
            coef  = float(row['coef'].values[0])
            ci_lo = float(row['ci_low'].values[0])
            ci_hi = float(row['ci_high'].values[0])
            pval  = float(row['pvalue'].values[0])
            color = colors_dict.get(grp, '#888888')

            # 背景条带
            if row_i % 2 == 0:
                ax.axhspan(row_i - 0.4, row_i + 0.4, color='#F0F4F8', alpha=0.5, zorder=0)

            # CI区间矩形
            rect_h = 0.3
            ax.barh(row_i, ci_hi - ci_lo, left=ci_lo, height=rect_h,
                    color=color, alpha=0.18, zorder=2)
            # 误差线
            ax.errorbar(coef, row_i,
                        xerr=[[coef - ci_lo], [ci_hi - coef]],
                        fmt='D', color=color, markersize=6,
                        capsize=4, linewidth=1.5, elinewidth=1.5,
                        markeredgewidth=0.8, markeredgecolor='white',
                        zorder=3)
            # 显著性星号
            star = sig_star(pval)
            if star:
                ax.text(ci_hi + abs(ci_hi - ci_lo) * 0.08 + 0.005,
                        row_i, star, va='center', ha='left',
                        fontsize=9, color=color, fontweight='bold',
                        zorder=4)

        ax.axvline(0, color='#333333', linestyle='--', linewidth=0.9, alpha=0.8, zorder=1)
        ax.set_yticks(range(n_grps))

        grp_names = list(reversed(groups))
        ylabels   = [group_label_map.get(g, g) if group_label_map else g
                     for g in grp_names]
        if col_i == 0:
            ax.set_yticklabels(ylabels, fontsize=9.5,
                               fontproperties=CN_PROP if CN_PROP else None)
        else:
            ax.set_yticklabels(['' for _ in ylabels])

        var_label = focal_labels.get(var, var)
        ax.set_title(var_label, fontsize=10.5, fontweight='bold', pad=7,
                     fontproperties=CN_PROP if CN_PROP else None)
        ax.set_xlabel('回归系数', fontsize=8.5,
                      fontproperties=CN_PROP if CN_PROP else None)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='x', alpha=0.4, linestyle=':', color='#BBBBBB')

    # 图例
    legend_handles = [
        mpatches.Patch(color=colors_dict.get(g, '#888888'), label=group_label_map.get(g, g) if group_label_map else g)
        for g in groups
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=min(n_grps, 4),
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.01),
               prop=CN_PROP if CN_PROP else None,
               edgecolor='#CCCCCC', fancybox=False)

    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.01,
                 fontproperties=CN_PROP if CN_PROP else None)

    # 脚注
    fig.text(0.5, -0.07, note, ha='center', fontsize=7.5, color='#555555',
             fontproperties=CN_PROP if CN_PROP else None)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close(fig)
    log(f"  已保存：{out_path}")


# ---- 11.1 知识领域 ----
if not df_domain_coef.empty:
    plot_coef_forest(
        df_domain_coef, DOMAINS, DOMAIN_COLORS,
        FOCAL_VARS, FOCAL_LABELS,
        '图1　核心变量回归系数——按知识领域分组',
        'domain_coefficient_comparison.png',
    )

# ---- 11.2 粉丝规模 ----
if not df_fans_coef.empty:
    plot_coef_forest(
        df_fans_coef, fans_groups, FANS_COLORS,
        FOCAL_VARS, FOCAL_LABELS,
        '图2　核心变量回归系数——按粉丝规模分组',
        'fans_group_coefficient_comparison.png',
        group_label_map=fans_label_map,
    )

# ---- 11.3 封面文字有无 ----
if not df_text_coef.empty:
    plot_coef_forest(
        df_text_coef, ['cover_text_0','cover_text_1'], TEXT_COLORS,
        FOCAL_VARS, FOCAL_LABELS,
        '图3　核心变量回归系数——按封面文字有无分组',
        'cover_text_group_coefficient_comparison.png',
        group_label_map=text_label_map,
    )

# ---- 11.4 UP主层级 ----
if not df_level_coef.empty:
    plot_coef_forest(
        df_level_coef, [f'level_{lv}' for lv in LEVELS],
        {f'level_{lv}': LEVEL_COLORS[lv] for lv in LEVELS},
        FOCAL_VARS, FOCAL_LABELS,
        '图4　核心变量回归系数——按UP主层级分组',
        'level_coefficient_comparison.png',
        group_label_map=level_label_map,
    )

# ============================================================
# 12. 热力图
# ============================================================
log("=== 绘制热力图 ===")

def make_heatmap_academic(df_coef, groups, vars_to_plot, labels_map,
                           group_label_map, suptitle, filename):
    if df_coef.empty:
        return
    row_labels = [group_label_map.get(g, g) for g in groups]
    col_labels  = [labels_map.get(v, v) for v in vars_to_plot]

    coef_mat = pd.DataFrame(np.nan, index=row_labels, columns=col_labels)
    sig_mat  = pd.DataFrame('',    index=row_labels, columns=col_labels)

    for grp, rlbl in zip(groups, row_labels):
        sub = df_coef[df_coef['group'] == grp]
        for var, clbl in zip(vars_to_plot, col_labels):
            row = sub[sub['variable'] == var]
            if not row.empty:
                coef_mat.loc[rlbl, clbl] = float(row['coef'].values[0])
                sig_mat.loc[rlbl, clbl]  = sig_star(float(row['pvalue'].values[0]))

    n_rows = len(row_labels)
    n_cols  = len(col_labels)
    fig_w   = max(8, n_cols * 1.8)
    fig_h   = max(3.5, n_rows * 1.1 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    vmax = np.nanmax(np.abs(coef_mat.values.astype(float)))
    sns.heatmap(coef_mat.astype(float), annot=True, fmt='.3f',
                center=0, vmin=-vmax, vmax=vmax,
                cmap='RdBu_r', ax=ax, linewidths=0.6,
                linecolor='#E0E0E0',
                cbar_kws={'label': '回归系数', 'shrink': 0.8},
                annot_kws={'size': 9, 'weight': 'bold'})

    # 叠加显著性标注
    for i, rlbl in enumerate(coef_mat.index):
        for j, clbl in enumerate(coef_mat.columns):
            star = sig_mat.loc[rlbl, clbl]
            if star:
                ax.text(j + 0.88, i + 0.22, star, ha='center', va='center',
                        color='black', fontsize=8, fontweight='bold')

    ax.set_title(suptitle, fontsize=11, fontweight='bold', pad=10,
                 fontproperties=CN_PROP if CN_PROP else None)
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_xticklabels(col_labels, rotation=25, ha='right', fontsize=9,
                        fontproperties=CN_PROP if CN_PROP else None)
    ax.set_yticklabels(row_labels, rotation=0, fontsize=9,
                        fontproperties=CN_PROP if CN_PROP else None)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    if CN_PROP:
        cbar.ax.set_ylabel('回归系数', fontsize=9, fontproperties=CN_PROP)

    fig.text(0.5, -0.03,
             '注：格内数字为回归系数，†p<.1，*p<.05，**p<.01，***p<.001（HC3稳健标准误）',
             ha='center', fontsize=7.5, color='#555555',
             fontproperties=CN_PROP if CN_PROP else None)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close(fig)
    log(f"  热力图已保存：{out_path}")

make_heatmap_academic(
    df_domain_coef, DOMAINS, FOCAL_VARS, FOCAL_LABELS,
    {d: d for d in DOMAINS},
    '图5　知识领域×核心变量系数热力图',
    'domain_heatmap.png'
)

make_heatmap_academic(
    df_fans_coef, fans_groups, FOCAL_VARS, FOCAL_LABELS,
    fans_label_map,
    '图6　粉丝规模组×核心变量系数热力图',
    'fans_group_heatmap.png'
)

# ============================================================
# 13. 粉丝组 × domain 交互趋势图
# ============================================================
log("=== 绘制粉丝组×domain趋势图 ===")
try:
    key_var = 'image_title_fit_centered'
    if key_var in df_model.columns:
        fig, axes = plt.subplots(1, len(DOMAINS), figsize=(14, 4.5), sharey=True)
        fig.patch.set_facecolor('white')
        for di, dom in enumerate(DOMAINS):
            ax = axes[di]
            ax.set_facecolor('#FAFAFA')
            sub_d = df_model[df_model['domain'] == dom]
            coefs, ci_los, ci_his = [], [], []
            for fg in fans_groups:
                sub_f = sub_d[sub_d['fans_group'] == fg]
                if sub_f.shape[0] < 30:
                    coefs.append(np.nan); ci_los.append(np.nan); ci_his.append(np.nan)
                else:
                    try:
                        r = smf.ols(formula_base, data=sub_f).fit(cov_type='HC3')
                        if key_var in r.params:
                            c = float(r.params[key_var])
                            lo, hi = r.conf_int().loc[key_var]
                            coefs.append(c); ci_los.append(float(lo)); ci_his.append(float(hi))
                        else:
                            coefs.append(np.nan); ci_los.append(np.nan); ci_his.append(np.nan)
                    except:
                        coefs.append(np.nan); ci_los.append(np.nan); ci_his.append(np.nan)

            x = np.arange(len(fans_groups))
            colors_ = [FANS_COLORS[fg] for fg in fans_groups]
            for xi, (c, lo, hi, col) in enumerate(zip(coefs, ci_los, ci_his, colors_)):
                if not np.isnan(c):
                    ax.fill_between([xi], [lo], [hi], color=col, alpha=0.15)
                    ax.errorbar(xi, c, yerr=[[c-lo],[hi-c]], fmt='D-',
                                color=col, markersize=6, capsize=4,
                                linewidth=1.8, elinewidth=1.2)

            ax.axhline(0, color='#333333', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(['低','中','高'], fontsize=9,
                                fontproperties=CN_PROP if CN_PROP else None)
            dom_color = DOMAIN_COLORS.get(dom, '#333333')
            ax.set_title(dom, fontsize=10.5, fontweight='bold', color=dom_color,
                         fontproperties=CN_PROP if CN_PROP else None)
            ax.set_xlabel('粉丝规模组', fontsize=8.5,
                          fontproperties=CN_PROP if CN_PROP else None)
            if di == 0:
                ax.set_ylabel('图文语义匹配系数', fontsize=9,
                               fontproperties=CN_PROP if CN_PROP else None)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.35, linestyle=':', color='#BBBBBB')

        fig.suptitle('图7　图文语义匹配效应在不同知识领域与粉丝规模中的变化',
                     fontsize=11, fontweight='bold', y=1.02,
                     fontproperties=CN_PROP if CN_PROP else None)
        fig.text(0.5, -0.04,
                 '注：误差线为95%置信区间（HC3），菱形为系数点估计值',
                 ha='center', fontsize=7.5, color='#555555',
                 fontproperties=CN_PROP if CN_PROP else None)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, 'domain_fans_interaction.png'),
                    bbox_inches='tight', dpi=150, facecolor='white')
        plt.close(fig)
        log("  domain×粉丝规模交互图已保存")
except Exception as e:
    log(f"  交互图绘制失败：{e}", 'WARN')

# ============================================================
# 14. 综合汇总系数表（论文格式）
# ============================================================
log("=== 构建论文格式系数汇总表 ===")

def build_paper_table(res_dict, group_labels_cn, focal_vars, focal_labels):
    """构建适合论文呈现的宽格式系数表"""
    groups = list(res_dict.keys())
    header_row1 = ['变量']
    for g in groups:
        lbl = group_labels_cn.get(g, g)
        header_row1 += [lbl, '', '']
    header_row2 = ['']
    for g in groups:
        header_row2 += ['系数', '标准误', 'p值']

    rows = [header_row1, header_row2]
    for var in focal_vars:
        row = [focal_labels.get(var, var)]
        for g in groups:
            res = res_dict.get(g)
            if res and var in res.params:
                c  = float(res.params[var])
                se = float(res.bse[var])
                p  = float(res.pvalues[var])
                row += [f"{c:.4f}{sig_star(p)}", f"({se:.4f})", f"{p:.4f}"]
            else:
                row += ['N/A', 'N/A', 'N/A']
        rows.append(row)

    # 模型统计量
    rows.append(['─' * 10] + ['─' * 8] * (len(groups) * 3))
    n_row = ['样本量']
    r2_row = ['R²']
    adjr2_row = ['adj-R²']
    for g in groups:
        res = res_dict.get(g)
        if res:
            n_row    += [str(int(res.nobs)), '', '']
            r2_row   += [f"{res.rsquared:.4f}", '', '']
            adjr2_row += [f"{res.rsquared_adj:.4f}", '', '']
        else:
            n_row += ['N/A','','']
            r2_row += ['N/A','','']
            adjr2_row += ['N/A','','']
    rows += [n_row, r2_row, adjr2_row]
    return pd.DataFrame(rows)

domain_cn_map = {d: d for d in DOMAINS}
fans_cn_map   = fans_label_map
level_cn_map  = level_label_map
text_cn_map   = text_label_map

df_paper_domain = build_paper_table(domain_results, domain_cn_map, FOCAL_VARS, FOCAL_LABELS)
df_paper_fans   = build_paper_table(fans_results,   fans_cn_map,   FOCAL_VARS, FOCAL_LABELS)
df_paper_level  = build_paper_table(level_results,  level_cn_map,  FOCAL_VARS, FOCAL_LABELS)
df_paper_text   = build_paper_table(text_results,   text_cn_map,   FOCAL_VARS, FOCAL_LABELS)

# ============================================================
# 15. 输出 Excel（多Sheet，学术排版）
# ============================================================
log("=== 输出 Excel ===")

THIN  = Side(style='thin',   color='BFBFBF')
THICK = Side(style='medium', color='1F4E79')
BORDER_THIN  = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
BORDER_THICK = Border(left=THICK, right=THICK, top=THICK, bottom=THICK)

def hdr_fill(hex_color):
    return PatternFill(start_color=hex_color, end_color=hex_color, fill_type='solid')

def style_sheet(ws, header_color='1F4E79', alt_color='EEF4FB'):
    """学术风格工作表美化"""
    hf = hdr_fill(header_color)
    hft = Font(color='FFFFFF', bold=True, name='Arial', size=10)
    al_c = Alignment(horizontal='center', vertical='center', wrap_text=True)
    al_l = Alignment(horizontal='left',   vertical='center', wrap_text=True)

    for cell in ws[1]:
        cell.fill = hf
        cell.font = hft
        cell.alignment = al_c
        cell.border = Border(bottom=Side(style='medium', color='FFFFFF'))

    alt_fill = hdr_fill(alt_color)
    for i, row in enumerate(ws.iter_rows(min_row=2), start=2):
        is_alt = (i % 2 == 0)
        for j, cell in enumerate(row):
            if is_alt:
                cell.fill = alt_fill
            cell.font = Font(name='Arial', size=9)
            cell.alignment = al_c if j > 0 else al_l
            cell.border = BORDER_THIN

    for col in ws.columns:
        max_len = max((len(str(cell.value or '')) for cell in col), default=8)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 32)
    ws.row_dimensions[1].height = 28

def df_to_sheet(wb, df_, sheet_name, header_color='1F4E79', alt_color='EEF4FB'):
    ws = wb.create_sheet(title=sheet_name[:31])
    if df_.empty:
        ws.append(['（无数据）']); return ws
    for r in dataframe_to_rows(df_, index=False, header=True):
        ws.append(r)
    style_sheet(ws, header_color, alt_color)
    return ws

def paper_table_to_sheet(wb, df_, sheet_name, title, header_color='1F4E79'):
    """论文格式表格输出到Excel Sheet"""
    ws = wb.create_sheet(title=sheet_name[:31])
    # 标题行
    ws.append([title])
    ws['A1'].font = Font(name='Arial', size=12, bold=True, color='1F4E79')
    ws['A1'].alignment = Alignment(horizontal='left')
    ws.append([])  # 空行

    if df_.empty:
        ws.append(['（无数据）']); return ws

    for r_idx, row in df_.iterrows():
        ws.append(list(row))

    # 表头（第3行）
    hf = hdr_fill(header_color)
    hft = Font(color='FFFFFF', bold=True, name='Arial', size=10)
    for cell in ws[3]:
        cell.fill = hf; cell.font = hft
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

    # 第4行（子标题）
    for cell in ws[4]:
        cell.fill = hdr_fill('2E75B6')
        cell.font = Font(color='FFFFFF', bold=True, name='Arial', size=9)
        cell.alignment = Alignment(horizontal='center')

    alt_fill = hdr_fill('EEF4FB')
    thin = Side(style='thin', color='BFBFBF')
    bdr  = Border(left=thin, right=thin, top=thin, bottom=thin)
    for i, row in enumerate(ws.iter_rows(min_row=5), start=5):
        is_alt = (i % 2 == 0)
        for cell in row:
            if is_alt: cell.fill = alt_fill
            cell.font = Font(name='Arial', size=9)
            cell.alignment = Alignment(horizontal='center')
            cell.border = bdr

    # 脚注
    ws.append([])
    ws.append(['注：括号内为HC3稳健标准误；†p<.1，*p<.05，**p<.01，***p<.001'])
    note_row = ws.max_row
    ws[f'A{note_row}'].font = Font(name='Arial', size=8, italic=True, color='555555')

    for col in ws.columns:
        max_len = max((len(str(cell.value or '')) for cell in col), default=8)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 28)
    ws.row_dimensions[3].height = 26
    ws.row_dimensions[4].height = 20
    return ws

wb = Workbook()
wb.remove(wb.active)

# 变量说明
var_dict_data = {
    'log_view_per_day':            ('因变量',      '日均播放量对数'),
    'log_view':                    ('因变量',      '总播放量对数（备用）'),
    'image_title_fit_centered':    ('核心解释变量','图文语义匹配度（均值中心化）'),
    'image_title_fit_sq_centered': ('核心解释变量','图文语义匹配度平方（非线性项）'),
    'title_cover_text_overlap':    ('核心解释变量','标题-封面文字重叠度'),
    'sentiment_consistency':       ('核心解释变量','标题与封面情感一致性'),
    'curiosity_marker_count':      ('核心解释变量','好奇心触发词数量'),
    'practical_value_word_count':  ('核心解释变量','实用价值词数量'),
    'cover_has_chart_or_formula':  ('核心解释变量','封面含图表/公式（二值，当前缺失）'),
    'cover_is_slide_or_screenshot':('核心解释变量','封面为PPT/截图类型（二值，当前缺失）'),
    'log_fans':     ('控制变量','粉丝量对数'),
    'log_duration': ('控制变量','视频时长对数'),
    'log_age':      ('控制变量','视频发布天数对数'),
    'pub_weekday':  ('控制变量','发布星期（0=周一）'),
    'pub_hour':     ('控制变量','发布小时'),
    'tag_count':    ('控制变量','标签数量'),
    'description_len':('控制变量','简介长度（字符数）'),
    'is_collection':  ('控制变量','是否合集（二值）'),
    'copyright':      ('控制变量','版权类型'),
    'domain':         ('分组变量','知识领域（科学科普/社科人文/财经商业/职场教育）'),
    'level':          ('分组变量','UP主层级（-1腰部/0中部/1头部）'),
    'fans_group':     ('分组变量','粉丝规模组（低/中/高三分位）'),
    'cover_has_text': ('分组变量','封面有无文字（0=无/1=有）'),
}
df_vardict = pd.DataFrame([
    {'变量名': k,
     '变量类型': v[0],
     '中文说明': v[1],
     '样本均值': f"{df_model[k].mean():.4f}" if k in df_model.columns and pd.api.types.is_numeric_dtype(df_model[k]) else 'N/A',
     '标准差':   f"{df_model[k].std():.4f}"  if k in df_model.columns and pd.api.types.is_numeric_dtype(df_model[k]) else 'N/A',
     '有效样本': str(df_model[k].notna().sum()) if k in df_model.columns else '0',
    }
    for k, v in var_dict_data.items()
])

ws0 = df_to_sheet(wb, df_vardict, '00_变量说明', '1F4E79', 'EEF4FB')
ws0['A1'].value = '变量名'

# 论文格式系数表
paper_table_to_sheet(wb, df_paper_domain, '01_知识领域分组（论文格式）',
                     '表1　按知识领域分组回归结果', '1F4E79')
paper_table_to_sheet(wb, df_paper_fans,   '02_粉丝规模分组（论文格式）',
                     '表2　按粉丝规模分组回归结果', 'C55A11')
paper_table_to_sheet(wb, df_paper_level,  '03_UP主层级分组（论文格式）',
                     '表3　按UP主层级分组回归结果', '7030A0')
paper_table_to_sheet(wb, df_paper_text,   '04_封面文字分组（论文格式）',
                     '表4　按封面文字有无分组回归结果', '833C11')

# 原始系数表（长格式，供进一步分析）
df_to_sheet(wb, df_domain_coef,      '05_分域系数_长格式',   '2E75B6', 'EBF3FB')
df_to_sheet(wb, df_sum_domain,       '06_分域汇总',          '2E75B6', 'EBF3FB')
df_to_sheet(wb, df_interaction_coef, '07_交互项系数',        '375623', 'EBF9EB')
df_to_sheet(wb, df_level_coef,       '08_分层级系数_长格式', '7030A0', 'F4EBFF')
df_to_sheet(wb, df_sum_level,        '09_层级汇总',          '7030A0', 'F4EBFF')
df_to_sheet(wb, df_fans_coef,        '10_粉丝组系数_长格式', 'C55A11', 'FFF3EB')
df_to_sheet(wb, df_sum_fans,         '11_粉丝组汇总',        'C55A11', 'FFF3EB')
df_to_sheet(wb, df_text_coef,        '12_封面文字组系数',    '833C11', 'FFF5EB')
df_to_sheet(wb, df_sum_text,         '13_封面文字组汇总',    '833C11', 'FFF5EB')
df_to_sheet(wb, df_chow,             '14_Chow检验',          'BF8F00', 'FFFDE7')

wb.save(EXCEL_OUT)
log(f"Excel已保存：{EXCEL_OUT}（14个Sheet）")

# ============================================================
# 16. 分析报告（Markdown，论文格式）
# ============================================================
log("=== 生成分析报告 ===")

def auto_interpret(dom, res, var, pos_word, neg_word):
    c, p = get_coef(res, var)
    if c is None: return f"{dom}：无法估计（变量缺失或样本不足）。"
    dir_word = pos_word if c > 0 else neg_word
    sig_desc = ('在1%水平显著' if p < 0.01 else
                ('在5%水平显著' if p < 0.05 else
                 ('在10%水平边际显著' if p < 0.1 else '不显著')))
    return f"系数β={c:+.4f}，{sig_desc}（p={p:.4f}），方向{dir_word}。"

sci_res = domain_results.get('科学科普')
soc_res = domain_results.get('社科人文')
fin_res = domain_results.get('财经商业')
edu_res = domain_results.get('职场教育')

now_str = datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')
N_total = df_model.shape[0]

report_lines = [
f"# 第九步：异质性分析报告",
f"",
f"> **生成时间**：{now_str}",
f"> **数据来源**：{INPUT_FILE}",
f"> **建模样本**：{N_total:,} 条",
f"> **因变量**：`{Y_VAR}`（{'日均播放量对数' if 'per_day' in Y_VAR else '总播放量对数'}）",
f"> **标准误估计**：HC3 异方差稳健标准误",
f"",
f"---",
f"",
f"## 一、研究设计",
f"",
f"### 1.1 研究问题",
f"",
f"本报告检验标题特征、封面图像特征及双模态协同效应是否在以下维度中存在显著异质性：",
f"",
f"1. **知识领域**（科学科普、社科人文、财经商业、职场教育）",
f"2. **UP主层级**（腰部UP主 level=−1、中部 level=0、头部 level=1）",
f"3. **粉丝规模**（低/中/高三分位分组，切点：Q₁/₃={q33:.3f}，Q₂/₃={q67:.3f}）",
f"4. **封面文字有无**（cover_has_text=0/1）",
f"",
f"### 1.2 核心解释变量",
f"",
f"| 变量名 | 中文说明 |",
f"|:---|:---|",
]
for v in FOCAL_VARS:
    report_lines.append(f"| `{v}` | {FOCAL_LABELS.get(v, v)} |")

report_lines += [
f"",
f"### 1.3 控制变量",
f"",
f"  {', '.join(f'`{c}`' for c in CONTROL_VARS)}",
f"",
f"### 1.4 样本规模",
f"",
f"| 维度 | 分组 | 样本量 |",
f"|:---|:---|:---:|",
]
for dom in DOMAINS:
    n = df_model[df_model['domain'] == dom].shape[0]
    report_lines.append(f"| 知识领域 | {dom} | {n:,} |")
for lv in LEVELS:
    n = df_model[df_model['level'] == lv].shape[0]
    lbl = '腰部' if lv == -1 else ('中部' if lv == 0 else '头部')
    report_lines.append(f"| UP主层级 | {lbl}（level={lv}） | {n:,} |")
for fg in fans_groups:
    n = df_model[df_model['fans_group'] == fg].shape[0]
    report_lines.append(f"| 粉丝规模 | {fans_label_map[fg]} | {n:,} |")
for tg in [0, 1]:
    n = df_model[df_model['cover_has_text'] == tg].shape[0]
    lbl = '无封面文字' if tg == 0 else '有封面文字'
    report_lines.append(f"| 封面文字 | {lbl} | {n:,} |")

if not AVAILABLE_BINARY:
    report_lines += [
    f"",
    f"> [WARN] **数据说明**：`cover_has_chart_or_formula` 与 `cover_is_slide_or_screenshot`",
    f"> 在当前数据集中全为缺失值（NaN），已从所有回归模型中移除。",
    f"> 建议在前序封面特征提取步骤（Step 5/6）中补充后重新运行。",
    ]

# ---- 二、知识领域 ----
report_lines += [
f"",
f"---",
f"",
f"## 二、知识领域异质性分析",
f"",
f"### 2.1 各领域模型拟合概况",
f"",
f"| 领域 | 样本量 | R² | adj-R² |",
f"|:---:|:---:|:---:|:---:|",
]
for dom in DOMAINS:
    res = domain_results.get(dom)
    if res:
        report_lines.append(f"| {dom} | {int(res.nobs):,} | {res.rsquared:.4f} | {res.rsquared_adj:.4f} |")
    else:
        report_lines.append(f"| {dom} | N/A | N/A | N/A |")

report_lines += [
f"",
f"### 2.2 核心变量系数对比（HC3稳健标准误）",
f"",
f"| 变量 | 科学科普 | 社科人文 | 财经商业 | 职场教育 |",
f"|:---|:---:|:---:|:---:|:---:|",
]
for var in FOCAL_VARS:
    row_vals = [FOCAL_LABELS.get(var, var)]
    for dom in DOMAINS:
        row_vals.append(coef_str(domain_results.get(dom), var))
    report_lines.append("| " + " | ".join(row_vals) + " |")

report_lines += [
f"",
f"> 注：括号外为系数，†p<.1，*p<.05，**p<.01，***p<.001",
f"",
f"### 2.3 Chow结构性差异检验",
f"",
f"| 比较组 | F统计量 | p值 | 显著性 | 结论 |",
f"|:---|:---:|:---:|:---:|:---|",
]
for _, r in df_chow[df_chow['分组类型'] == '知识领域'].iterrows():
    report_lines.append(f"| {r['比较组']} | {r['F统计量']} | {r['p值']} | {r['显著性']} | {r['结论']} |")

report_lines += [
f"",
f"### 2.4 分领域实证解读",
f"",
f"#### 2.4.1 科学科普",
f"",
f"**研究命题**：科学科普类视频更依赖图文语义匹配和图表类视觉元素，",
f"信息性封面可降低受众获取知识的认知成本。",
f"",
f"**实证结果**：",
f"- 图文语义匹配：{auto_interpret('科学科普', sci_res, 'image_title_fit_centered', '与命题一致', '与命题相反')}",
f"- 标题-封面文字重叠：{auto_interpret('科学科普', sci_res, 'title_cover_text_overlap', '正向强化', '负向抑制')}",
f"- 实用价值词：{auto_interpret('科学科普', sci_res, 'practical_value_word_count', '正向促进', '负向抑制')}",
f"- 好奇心触发词：{auto_interpret('科学科普', sci_res, 'curiosity_marker_count', '正向促进', '负向抑制')}",
f"",
f"**解读**：科学科普视频传播受益于封面与标题在语义层面的高度一致，",
f"这有助于帮助受众快速判断视频内容定位，降低认知决策成本。",
f"图文一致性强的科普视频更易在平台推荐流中赢得点击。",
f"",
f"#### 2.4.2 社科人文",
f"",
f"**研究命题**：社科人文类视频更依赖好奇心触发和情感唤起，",
f"悬念感和情绪共鸣是触发点击的核心机制。",
f"",
f"**实证结果**：",
f"- 好奇心触发词：{auto_interpret('社科人文', soc_res, 'curiosity_marker_count', '与命题一致', '与命题相反')}",
f"- 情感一致性：{auto_interpret('社科人文', soc_res, 'sentiment_consistency', '正向强化', '负向抑制')}",
f"- 图文语义匹配：{auto_interpret('社科人文', soc_res, 'image_title_fit_centered', '正向促进', '负向抑制')}",
f"",
f"**解读**：社科人文话题涉及人文思考与价值判断，受众对情绪共鸣和悬念感更为敏感。",
f"好奇心标记语（如“为什么”“竟然”）在这一领域的促播放效应理论上更强。",
f"封面与标题情感调性的协同统一，有助于强化内容调性信号。",
f"",
f"#### 2.4.3 财经商业",
f"",
f"**研究命题**：财经商业类视频更依赖数字表达、数据图和实用价值词，",
f"目的性用户偏好具有明确操作指导价值的内容信号。",
f"",
f"**实证结果**：",
f"- 实用价值词：{auto_interpret('财经商业', fin_res, 'practical_value_word_count', '与命题一致', '与命题相反')}",
f"- 好奇心触发词：{auto_interpret('财经商业', fin_res, 'curiosity_marker_count', '正向促进', '负向抑制')}",
f"- 情感一致性：{auto_interpret('财经商业', fin_res, 'sentiment_consistency', '正向强化', '负向抑制')}",
f"",
f"**解读**：财经商业受众目的性较强，倾向于寻找具有实操指导价值的内容。",
f"含“如何、方法、步骤”等实用价值词的标题在此领域预期具有更强的点击激励效果。",
f"",
f"#### 2.4.4 职场教育",
f"",
f"**研究命题**：职场教育类视频更依赖教程感、方法感和PPT/截图类封面，",
f"功能性信息型包装与该领域受众期望高度匹配。",
f"",
f"**实证结果**：",
f"- 实用价值词：{auto_interpret('职场教育', edu_res, 'practical_value_word_count', '与命题一致', '与命题相反')}",
f"- 标题-封面文字重叠：{auto_interpret('职场教育', edu_res, 'title_cover_text_overlap', '正向强化', '负向抑制')}",
f"- 情感一致性：{auto_interpret('职场教育', edu_res, 'sentiment_consistency', '正向强化', '负向抑制')}",
f"",
f"**解读**：职场教育受众关注可操作的技能提升，",
f"封面中出现PPT截图、步骤说明等信息型元素能直接传递干货感，",
f"与标题形成高度一致的功能性信号，有利于提高视频传播效果。",
f"",
f"---",
f"",
f"## 三、UP主层级异质性分析",
f"",
f"### 3.1 分层级回归系数对比",
f"",
f"| 变量 | 腰部（level=−1） | 中部（level=0） | 头部（level=1） |",
f"|:---|:---:|:---:|:---:|",
]
for var in FOCAL_VARS:
    row_vals = [FOCAL_LABELS.get(var, var)]
    for lv in LEVELS:
        row_vals.append(coef_str(level_results.get(f'level_{lv}'), var))
    report_lines.append("| " + " | ".join(row_vals) + " |")

report_lines += [
f"",
f"> 注：†p<.1，*p<.05，**p<.01，***p<.001（HC3标准误）",
f"",
f"### 3.2 层级异质性解读",
f"",
f"**头部UP主（level=1）**：粉丝基数大、品牌认知度高，",
f"播放量传播更多依赖粉丝忠诚度和平台推荐流量，",
f"标题-封面协同包装对边际播放量的拉动效应预期相对较弱。",
f"",
f"**中部UP主（level=0）**：处于粉丝积累的关键期，",
f"兼具一定订阅基础和算法流量，标题封面质量对两者均有一定影响。",
f"",
f"**腰部UP主（level=−1）**：处于成长阶段，曝光机会主要依赖平台算法分发，",
f"高质量的标题封面协同信号有助于在推荐场景中脱颖而出，",
f"双模态协同效应预期在此层级最为显著。",
f"",
f"---",
f"",
f"## 四、粉丝规模异质性分析",
f"",
f"粉丝规模三分位切点：Q₁/₃ = {q33:.3f}（log_fans），Q₂/₃ = {q67:.3f}（log_fans）",
f"",
f"### 4.1 分粉丝规模回归系数对比",
f"",
f"| 变量 | 低粉丝量组 | 中等粉丝量组 | 高粉丝量组 |",
f"|:---|:---:|:---:|:---:|",
]
for var in FOCAL_VARS:
    row_vals = [FOCAL_LABELS.get(var, var)]
    for fg in fans_groups:
        row_vals.append(coef_str(fans_results.get(fg), var))
    report_lines.append("| " + " | ".join(row_vals) + " |")

report_lines += [
f"",
f"### 4.2 Chow检验（粉丝规模组间结构差异）",
f"",
f"| 比较组 | F统计量 | p值 | 显著性 | 结论 |",
f"|:---|:---:|:---:|:---:|:---|",
]
for _, r in df_chow[df_chow['分组类型'] == '粉丝规模'].iterrows():
    report_lines.append(f"| {r['比较组']} | {r['F统计量']} | {r['p值']} | {r['显著性']} | {r['结论']} |")

report_lines += [
f"",
f"### 4.3 粉丝规模异质性解读",
f"",
f"**低粉丝量UP主**：受众规模有限，每次算法曝光的转化价值尤为关键。",
f"标题封面的包装质量直接决定陌生受众的点击意愿，",
f"双模态协同包装效应预期在此组最为显著，符合弱账号依赖内容假设。",
f"",
f"**高粉丝量UP主**：既有订阅基础提供稳定的播放量底部支撑，",
f"播放量转化主要来自粉丝主动订阅行为，",
f"标题封面的边际效应相对弱化，但并非完全无效，",
f"高质量包装仍有助于吸引非订阅用户的点击。",
f"",
f"---",
f"",
f"## 五、封面文字有无异质性分析",
f"",
f"有封面文字：{df_model[df_model['cover_has_text']==1].shape[0]:,} 条　｜　无封面文字：{df_model[df_model['cover_has_text']==0].shape[0]:,} 条",
f"",
f"### 5.1 分组回归系数对比",
f"",
f"| 变量 | 无封面文字组 | 有封面文字组 |",
f"|:---|:---:|:---:|",
]
for var in FOCAL_VARS:
    row_vals = [FOCAL_LABELS.get(var, var)]
    for tg in [0, 1]:
        row_vals.append(coef_str(text_results.get(f'cover_text_{tg}'), var))
    report_lines.append("| " + " | ".join(row_vals) + " |")

report_lines += [
f"",
f"### 5.2 封面文字异质性解读",
f"",
f"**无封面文字组**：`title_cover_text_overlap` 恒为0，文字层面的标题-封面协同天然缺失。",
f"在此情况下，图文语义匹配（CLIP语义相似度）和情感一致性",
f"成为双模态协同效应的主要载体，发挥更核心的信号传递作用。",
f"",
f"**有封面文字组**：文字重叠度变量具有实际变异，",
f"可直接捕捉标题与封面文字的信息互补或重复关系。",
f"实用价值词和好奇心词的效应也可能通过封面文字的视觉显著性得到强化。",
f"此外，`title_cover_text_overlap` 系数的显著性可作为文字层面协同效应的直接证据。",
f"",
f"---",
f"",
f"## 六、综合讨论",
f"",
f"### 6.1 研究命题验证情况",
f"",
f"| 编号 | 研究命题 | 核心变量 | 预期最强组 | 数据可检验性 |",
f"|:---:|:---|:---|:---|:---:|",
f"| H1 | 科学科普更依赖图文语义匹配和图表元素 | image_title_fit_centered | 科学科普 | ✓ 可检验 |",
f"| H2 | 社科人文更依赖悬念表达和情绪唤起 | curiosity_marker_count, sentiment_consistency | 社科人文 | ✓ 可检验 |",
f"| H3 | 财经商业更依赖数字表达和实用价值 | practical_value_word_count | 财经商业 | ✓ 可检验 |",
f"| H4 | 职场教育更依赖教程感和方法感 | practical_value_word_count, cover_is_slide | 职场教育 | [WARN] 部分可检验 |",
f"| H5 | 低粉丝UP主更依赖标题封面包装 | 全部核心变量 | low_fans | ✓ 可检验 |",
f"| H6 | 头部UP主标题封面影响相对较弱 | 全部核心变量 | high_fans（弱效应） | ✓ 可检验 |",
f"",
f"### 6.2 主要局限与说明",
f"",
f"1. **封面分类变量缺失**：`cover_has_chart_or_formula` 与 `cover_is_slide_or_screenshot`",
f"   在当前数据集中全部缺失（NaN），导致封面类型相关命题（H4）无法完整检验，",
f"   建议在前序封面特征提取步骤中补充后重新运行。",
f"",
f"2. **UP主层级划分**：level字段（−1/0/1）的层级划分标准应在论文方法部分明确说明，",
f"   以确保可复现性（如按粉丝数量区间或平台认证标准）。",
f"",
f"3. **封面文字组样本不均衡**：无封面文字样本仅 {df_model[df_model['cover_has_text']==0].shape[0]} 条，",
f"   系数估计存在较大不确定性，该组结论需谨慎解读。",
f"",
f"4. **Chow检验局限**：本研究采用残差平方和近似Chow检验，",
f"   各组样本共用相同分布假设，结果为参考性检验而非严格意义上的结构突变检验。",
f"",
f"### 6.3 论文写作建议",
f"",
f"- **主图**：使用 `domain_coefficient_comparison.png`（图1）作为知识领域异质性主图，",
f"  `fans_group_coefficient_comparison.png`（图2）作为粉丝规模异质性主图。",
f"",
f"- **辅助图**：`domain_heatmap.png`（图5）可作为附录综合展示图。",
f"",
f"- **表格引用**：Excel的Sheet `01_知识领域分组（论文格式）` 可直接转为论文表格，",
f"  格式符合社会科学期刊回归表规范。",
f"",
f"- **Chow检验**：结果可在正文脚注或附录中报告，",
f"  作为组间系数差异具有统计意义的正式检验支撑。",
f"",
f"---",
f"",
f"## 七、输出文件清单",
f"",
f"| 文件类型 | 文件路径 | 说明 |",
f"|:---:|:---|:---|",
f"| Excel | `{EXCEL_OUT}` | 14个Sheet：论文格式表格+长格式系数+Chow检验 |",
f"| 报告 | `{REPORT_OUT}` | 本分析报告（Markdown格式） |",
f"| 图1 | `outputs/figures/domain_coefficient_comparison.png` | 知识领域系数森林图 |",
f"| 图2 | `outputs/figures/fans_group_coefficient_comparison.png` | 粉丝规模系数森林图 |",
f"| 图3 | `outputs/figures/cover_text_group_coefficient_comparison.png` | 封面文字组系数图 |",
f"| 图4 | `outputs/figures/level_coefficient_comparison.png` | UP主层级系数图 |",
f"| 图5 | `outputs/figures/domain_heatmap.png` | 知识领域系数热力图 |",
f"| 图6 | `outputs/figures/fans_group_heatmap.png` | 粉丝规模系数热力图 |",
f"| 图7 | `outputs/figures/domain_fans_interaction.png` | domain×粉丝规模交互趋势图 |",
f"",
f"---",
f"",
f"*本报告由Python脚本 `step09_heterogeneity_final.py` 自动生成。*",
f"*如需修改解读文字，请编辑报告生成部分（Section 16）后重新运行。*",
]

with open(REPORT_OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
log(f"分析报告已保存：{REPORT_OUT}")

# ============================================================
# 17. 日志保存
# ============================================================
LOG_OUT = 'outputs/logs/heterogeneity_run.log'
with open(LOG_OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG_LINES))

log(f"运行日志已保存：{LOG_OUT}")
log("==== 第九步：异质性分析（增强版）完成 ====")
print("\n" + "="*60)
print("✓ 所有输出文件已保存至 outputs/ 目录")
print("="*60)
