"""
第七步：相关性与多重共线性检验
研究题目：标题和封面双模态协同对社交媒体平台视频传播的影响——以B站知识类视频为例

依赖包：
    pip install pandas numpy scipy matplotlib seaborn openpyxl statsmodels scikit-learn

使用说明：
    将本脚本与 modeling_dataset.csv 放在同一文件夹下，直接运行即可。
    输出文件（Excel、图片、报告）会生成在脚本所在目录的 outputs/ 子文件夹中。
"""

import os, sys, warnings, logging
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── 路径配置（自动适配本地 / 服务器环境）────────────────────────────────────────
# 脚本所在目录（Windows 本地、Linux 服务器均适用）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(os.getcwd(), "outputs", "modeling_dataset.csv")

# 若脚本目录找不到数据，再尝试常见备用路径
_FALLBACKS = [
    os.path.join(SCRIPT_DIR, "..", "modeling_dataset.csv"),
    "/mnt/user-data/uploads/modeling_dataset.csv",
]
if not os.path.exists(INPUT_FILE):
    for _fb in _FALLBACKS:
        if os.path.exists(_fb):
            INPUT_FILE = os.path.abspath(_fb)
            break

OUT_DIR  = os.path.join(os.getcwd(), "outputs")
FIG_DIR  = os.path.join(OUT_DIR, "figures")
LOG_DIR  = os.path.join(OUT_DIR, "logs")
XLSX_OUT = os.path.join(OUT_DIR, "correlation_vif_results.xlsx")
LOG_MD   = os.path.join(LOG_DIR, "correlation_vif_report.md")
FIG_FULL = os.path.join(FIG_DIR, "correlation_heatmap_full.png")
FIG_CORE = os.path.join(FIG_DIR, "correlation_heatmap_core.png")

for d in [OUT_DIR, FIG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 中文字体 ──────────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
_cands = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
_font  = next((f for c in _cands for f in fm.fontManager.ttflist
               if c.lower() in f.name.lower()), None)
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   [_font.name if _font else "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
})

# ── 日志 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "step7_runtime.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8", filemode="w",
)
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
log.info("=== 第七步：相关性与多重共线性检验 开始 ===")

# ══════════════════════════════════════════════════════════════════════════════
# 1. 读取数据
# ══════════════════════════════════════════════════════════════════════════════
if not os.path.exists(INPUT_FILE):
    log.error(f"找不到数据文件：{INPUT_FILE}")
    log.error("请将 modeling_dataset.csv 与本脚本放在同一目录下后重新运行。")
    sys.exit(1)

log.info(f"读取：{INPUT_FILE}")
df_raw = pd.read_csv(INPUT_FILE, low_memory=False)
log.info(f"原始维度：{df_raw.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. 变量定义与中文别名
# ══════════════════════════════════════════════════════════════════════════════
DV    = ["log_view","log_view_per_day","recognition","participation",
         "communication_effect","relative_view"]
CTRL  = ["log_fans","log_age","log_duration","tag_count","description_len"]
TITLE = ["title_char_len","title_question_type","title_exclamation_type",
         "title_has_number","title_sentiment_intensity",
         "curiosity_marker_count","practical_value_word_count"]
COVER = ["image_brightness","image_contrast","image_colorfulness",
         "visual_complexity","cover_has_text","cover_text_area_ratio",
         "face_presence","cover_has_chart_or_formula","cover_is_slide_or_screenshot"]
BIMOD = ["image_title_fit","image_title_fit_sq","title_cover_text_overlap",
         "sentiment_consistency","sentiment_conflict",
         "both_high_arousal","semantic_text_interaction"]

ALL_VARS = DV + CTRL + TITLE + COVER + BIMOD

LABEL = {
    "log_view":"对数总播放量","log_view_per_day":"对数日均播放量",
    "recognition":"认知传播效果","participation":"参与传播效果",
    "communication_effect":"综合传播效果","relative_view":"相对播放量",
    "log_fans":"对数粉丝数","log_age":"对数视频年龄（天）",
    "log_duration":"对数视频时长","tag_count":"标签数量",
    "description_len":"简介字符数","title_char_len":"标题字符长度",
    "title_question_type":"标题疑问句型","title_exclamation_type":"标题感叹句型",
    "title_has_number":"标题含数字","title_sentiment_intensity":"标题情感强度",
    "curiosity_marker_count":"好奇触发词数",
    "practical_value_word_count":"实用价值词数",
    "image_brightness":"封面亮度","image_contrast":"封面对比度",
    "image_colorfulness":"封面色彩丰富度","visual_complexity":"视觉复杂度",
    "cover_has_text":"封面含文字","cover_text_area_ratio":"封面文字占比",
    "face_presence":"封面含人脸","cover_has_chart_or_formula":"封面含图表/公式",
    "cover_is_slide_or_screenshot":"封面为截图/幻灯片",
    "image_title_fit":"图文语义匹配度","image_title_fit_sq":"图文匹配度平方项",
    "title_cover_text_overlap":"标封文字重叠度",
    "sentiment_consistency":"情感一致性","sentiment_conflict":"情感冲突",
    "both_high_arousal":"双高唤起","semantic_text_interaction":"语义—文字交互",
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. 数据质量检查 & 筛选可用变量
# ══════════════════════════════════════════════════════════════════════════════
issues = []
AVAIL  = []
for v in ALL_VARS:
    if v not in df_raw.columns:
        log.warning(f"[列缺失] {v} 不在数据中")
        issues.append((v, "列不存在", "-"))
        continue
    miss     = df_raw[v].isna().sum()
    miss_pct = miss / len(df_raw)
    if miss_pct >= 0.5:
        log.warning(f"[严重缺失] {v}：{miss} ({miss_pct:.0%})，跳过此变量")
        issues.append((v, f"严重缺失 {miss_pct:.0%}", "已排除出分析"))
    else:
        AVAIL.append(v)
        if miss > 0:
            log.info(f"[少量缺失] {v}：{miss} 缺失，将用中位数填充")
            issues.append((v, f"缺失 {miss} 个", "中位数填充"))

log.info(f"可用变量数：{len(AVAIL)}")

df = df_raw[AVAIL].copy()
for v in AVAIL:
    if df[v].isna().sum() > 0:
        df[v] = df[v].fillna(df[v].median())

df_c = df.dropna()
N    = len(df_c)
log.info(f"分析样本量：{N}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. 相关系数矩阵（Pearson + Spearman）
# ══════════════════════════════════════════════════════════════════════════════
def corr_pval(data, method="pearson"):
    cols = list(data.columns)
    n    = len(cols)
    R    = pd.DataFrame(np.nan, index=cols, columns=cols)
    P    = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i in range(n):
        R.iloc[i, i] = 1.0
        P.iloc[i, i] = 0.0
        for j in range(i + 1, n):
            x   = data.iloc[:, i].values
            y   = data.iloc[:, j].values
            msk = ~(np.isnan(x) | np.isnan(y))
            if msk.sum() < 10:
                continue
            fn  = stats.pearsonr if method == "pearson" else stats.spearmanr
            r, p = fn(x[msk], y[msk])
            R.iloc[i, j] = R.iloc[j, i] = r
            P.iloc[i, j] = P.iloc[j, i] = p
    return R.astype(float), P.astype(float)

log.info("计算 Pearson 相关系数…")
PR, PP = corr_pval(df_c[AVAIL], "pearson")
log.info("计算 Spearman 相关系数…")
SR, SP = corr_pval(df_c[AVAIL], "spearman")
log.info("相关系数计算完成。")

# ══════════════════════════════════════════════════════════════════════════════
# 5. 热力图
# ══════════════════════════════════════════════════════════════════════════════
def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

def plot_heatmap(corr_df, pval_df, title, fpath, threshold=0.3, base=0.52):
    vars_ = list(corr_df.index)
    n     = len(vars_)
    fs    = max(10, n * base + 2)
    fig, ax = plt.subplots(figsize=(fs, fs * 0.88))

    mask_upper = np.triu(np.ones((n, n), bool), k=0)

    ann = np.full((n, n), "", dtype=object)
    for i in range(n):
        for j in range(i):
            r = corr_df.iloc[i, j]
            if abs(r) > threshold:
                ann[i, j] = f"{r:.2f}{sig_stars(pval_df.iloc[i, j])}"

    lbls  = [LABEL.get(v, v) for v in vars_]
    fsize = max(5, int(130 / n))

    sns.heatmap(corr_df, mask=mask_upper, annot=ann, fmt="",
                cmap="RdYlBu_r", vmin=-1, vmax=1, center=0,
                linewidths=0.3, linecolor="white",
                xticklabels=lbls, yticklabels=lbls,
                cbar_kws={"shrink": 0.6, "label": "相关系数"},
                annot_kws={"size": fsize}, ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=max(6, fsize))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=max(6, fsize))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, -0.01,
             f"注：* p<0.05，** p<0.01，*** p<0.001；仅标注 |r|>{threshold} 的系数；N={N}",
             ha="center", fontsize=9, color="#555555")
    plt.tight_layout()
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    log.info(f"热力图保存：{fpath}")

log.info("绘制全变量热力图…")
plot_heatmap(PR, PP,
    "图X  全变量 Pearson 相关系数矩阵",
    FIG_FULL, threshold=0.3, base=0.52)

CORE_AVAIL = [v for v in (TITLE + COVER + BIMOD) if v in AVAIL]
log.info("绘制核心解释变量热力图…")
plot_heatmap(PR.loc[CORE_AVAIL, CORE_AVAIL],
             PP.loc[CORE_AVAIL, CORE_AVAIL],
    "图X  核心解释变量 Pearson 相关系数矩阵（标题·封面·双模态）",
    FIG_CORE, threshold=0.25, base=0.65)

# ══════════════════════════════════════════════════════════════════════════════
# 6. VIF 多重共线性
# ══════════════════════════════════════════════════════════════════════════════
log.info("开始 VIF 计算…")

df_vif_base = df_c.copy()

if "image_title_fit" in df_vif_base.columns:
    df_vif_base["image_title_fit_c"]    = (df_vif_base["image_title_fit"]
                                            - df_vif_base["image_title_fit"].mean())
    df_vif_base["image_title_fit_c_sq"] = df_vif_base["image_title_fit_c"] ** 2
    log.info("image_title_fit 已中心化，重建平方项。")

CONT_VIF = [v for v in [
    "log_fans","log_age","log_duration","tag_count","description_len",
    "title_char_len","title_sentiment_intensity",
    "curiosity_marker_count","practical_value_word_count",
    "image_brightness","image_contrast","image_colorfulness",
    "visual_complexity","cover_text_area_ratio",
    "image_title_fit_c","image_title_fit_c_sq",
    "title_cover_text_overlap","semantic_text_interaction",
] if v in df_vif_base.columns]

BIN_VIF = [v for v in [
    "title_question_type","title_exclamation_type","title_has_number",
    "cover_has_text","face_presence",
    "sentiment_consistency","sentiment_conflict","both_high_arousal",
] if v in df_vif_base.columns]

CAT_DFS = []
for cat in ["domain", "level", "pub_month"]:
    if cat not in df_raw.columns:
        continue
    raw_cat = df_raw.loc[df_vif_base.index, cat]
    d = pd.get_dummies(raw_cat, prefix=cat, drop_first=True).astype(float)
    d = d.loc[:, d.std() > 0]
    if d.shape[1] > 0:
        CAT_DFS.append(d)
        log.info(f"Dummy {cat}：{d.shape[1]} 列 → {list(d.columns)}")
    else:
        log.warning(f"Dummy {cat}：编码后无有效列，跳过。")

scaler     = StandardScaler()
df_cont_z  = pd.DataFrame(
    scaler.fit_transform(df_vif_base[CONT_VIF]),
    columns=[v + "_z" for v in CONT_VIF],
    index=df_vif_base.index,
)

parts      = [df_cont_z, df_vif_base[BIN_VIF]] + CAT_DFS
df_vif_mat = pd.concat(parts, axis=1).dropna()

const_cols = [c for c in df_vif_mat.columns if df_vif_mat[c].nunique() <= 1]
if const_cols:
    log.warning(f"删除常数列：{const_cols}")
    df_vif_mat.drop(columns=const_cols, inplace=True)

log.info(f"VIF 矩阵：{df_vif_mat.shape[0]} 行 × {df_vif_mat.shape[1]} 列")

X_arr    = df_vif_mat.values.astype(float)
vif_rows = []
for i, col in enumerate(df_vif_mat.columns):
    try:
        v_val = variance_inflation_factor(X_arr, i)
    except Exception as e:
        v_val = np.nan
        log.warning(f"[VIF 失败] {col}：{e}")

    orig_name = col.replace("_z", "").replace("_c_sq", "_sq").replace("_c", "")

    status = ("✓ 可接受"  if (not np.isnan(v_val) and v_val < 5)
              else "[WARN] 需关注"  if (not np.isnan(v_val) and v_val < 10)
              else "✗ 严重共线" if not np.isnan(v_val)
              else "N/A")

    vif_rows.append({
        "变量（英文）": col,
        "中文含义":    LABEL.get(orig_name, orig_name),
        "VIF":         round(v_val, 3) if not np.isnan(v_val) else "N/A",
        "判断":        status,
    })

df_vif_res = (pd.DataFrame(vif_rows)
              .sort_values("VIF",
                           key=lambda s: pd.to_numeric(s, errors="coerce"),
                           ascending=False)
              .reset_index(drop=True))

log.info("\n" + df_vif_res.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 7. 高度相关变量对
# ══════════════════════════════════════════════════════════════════════════════
def hi_pairs(corr_df, thr):
    cols = list(corr_df.columns)
    out  = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_df.iloc[i, j]
            if abs(r) >= thr:
                out.append((cols[i], cols[j], round(r, 4)))
    return sorted(out, key=lambda x: abs(x[2]), reverse=True)

pairs50 = hi_pairs(PR.loc[AVAIL, AVAIL], 0.5)
pairs70 = hi_pairs(PR.loc[AVAIL, AVAIL], 0.7)
pairs30 = hi_pairs(PR.loc[AVAIL, AVAIL], 0.3)

pdict = {(a, b): r for a, b, r in pairs50}
def get_r(a, b): return pdict.get((a, b), pdict.get((b, a), None))

# ══════════════════════════════════════════════════════════════════════════════
# 8. Markdown 报告
# ══════════════════════════════════════════════════════════════════════════════
severe_vif = df_vif_res[df_vif_res["判断"].str.contains("严重", na=False)]
warn_vif   = df_vif_res[df_vif_res["判断"].str.contains("需关注", na=False)]

md = []
md.append("# 第七步：相关性与多重共线性检验报告")
md.append(f"\n> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}  \n> 样本量 N = {N}")
md.append("\n---\n")

md.append("## 一、数据质量")
if issues:
    md.append("\n| 变量 | 问题 | 处理方式 |")
    md.append("|------|------|---------|")
    for v, iss, hdl in issues:
        md.append(f"| `{v}` | {iss} | {hdl} |")
else:
    md.append("\n所有目标变量完整，无缺失。")

md.append("\n---\n")
md.append("## 二、高度相关变量对（|r| ≥ 0.5）")
md.append("\n| 变量A | 含义A | 变量B | 含义B | Pearson r |")
md.append("|-------|-------|-------|-------|-----------|")
for a, b, r in pairs50:
    md.append(f"| `{a}` | {LABEL.get(a,a)} | `{b}` | {LABEL.get(b,b)} | {r:.4f} |")
if not pairs50:
    md.append("| — | — | — | — | — |")

md.append("\n---\n")
md.append("## 三、极高相关变量对（|r| ≥ 0.7）")
md.append("\n| 变量A | 变量B | r |")
md.append("|-------|-------|---|")
for a, b, r in pairs70:
    md.append(f"| `{a}` | `{b}` | {r:.4f} |")
if not pairs70:
    md.append("| — | — | — |")

md.append("\n---\n")
md.append("## 四、VIF 多重共线性检验")
md.append("\n| 变量（英文） | 中文含义 | VIF | 判断 |")
md.append("|-------------|---------|-----|------|")
for _, row in df_vif_res.iterrows():
    md.append(f"| `{row['变量（英文）']}` | {row['中文含义']} | {row['VIF']} | {row['判断']} |")

md.append("\n---\n")
md.append("## 五、自动诊断与建议")

md.append("\n### 5.1 高度相关变量说明\n")
notes = []
r_tc = get_r("title_char_len", "title_word_count")
if r_tc and abs(r_tc) >= 0.5:
    notes.append(f"- **`title_char_len` ↔ `title_word_count`（r={r_tc:.3f}）**：两者高度相关，"
                 "主回归**保留 `title_char_len`（标题字符长度）**，"
                 "`title_word_count` 进入稳健性检验替代模型。")
r_fit = get_r("image_title_fit", "image_title_fit_sq")
if r_fit and abs(r_fit) >= 0.5:
    notes.append(f"- **`image_title_fit` ↔ `image_title_fit_sq`（r={r_fit:.3f}）**："
                 "已对原变量进行均值中心化后重建平方项（`image_title_fit_c_sq`），"
                 "中心化后共线性显著降低，VIF 见上表。")
r_ct = get_r("cover_has_text", "cover_text_area_ratio")
if r_ct and abs(r_ct) >= 0.5:
    notes.append(f"- **`cover_has_text` ↔ `cover_text_area_ratio`（r={r_ct:.3f}）**："
                 "封面是否含文字（二值）与封面文字面积占比（连续）高度相关。"
                 "主模型**保留 `cover_text_area_ratio`**，`cover_has_text` 放入稳健性模型。")
r_sc = get_r("sentiment_consistency", "sentiment_conflict")
if r_sc is not None:
    notes.append(f"- **`sentiment_consistency` ↔ `sentiment_conflict`（r={r_sc:.3f}）**："
                 "两者按构造规则互斥，不宜同时进入同一方程。"
                 "主模型保留 `sentiment_consistency`，`sentiment_conflict` 用于对立假设检验或分组回归。")
r_rp = get_r("recognition", "participation")
if r_rp and abs(r_rp) >= 0.5:
    notes.append(f"- **`recognition` ↔ `participation`（r={r_rp:.3f}）**："
                 "两者均为因变量，相关较高反映共同驱动因素，属正常现象；"
                 "回归时分别作为因变量独立建模，无需删除。")

for a, b, r in pairs50:
    if a in DV and b in DV: continue
    if a in DV or b in DV: continue
    pair_str = f"`{a}` ↔ `{b}`"
    already  = any(pair_str in n for n in notes)
    if not already:
        notes.append(f"- **{pair_str}（r={r:.3f}）**："
                     f"{LABEL.get(a,a)} 与 {LABEL.get(b,b)} 存在高度相关，"
                     "在主模型中建议择一保留，另一变量进入稳健性模型。")

if notes:
    md += notes
else:
    md.append("- 核心解释变量间高度相关对较少，整体多重共线性状况较好。")

md.append("\n### 5.2 VIF 严重共线（≥10）\n")
if len(severe_vif) > 0:
    for _, row in severe_vif.iterrows():
        md.append(f"- **`{row['变量（英文）']}`（VIF={row['VIF']}）**："
                  "存在严重多重共线性，建议从主模型中移除，或与其高度相关变量择一。")
else:
    md.append("- 无 VIF ≥ 10 的严重共线变量，整体共线性状况良好。 ✓")

md.append("\n### 5.3 需关注（5≤VIF<10）\n")
if len(warn_vif) > 0:
    for _, row in warn_vif.iterrows():
        md.append(f"- `{row['变量（英文）']}`（VIF={row['VIF']}）："
                  "存在一定共线性，可保留但须在论文中说明并进行稳健性检验。")
else:
    md.append("- 无 VIF 在 5–10 之间的变量。")

md.append("\n### 5.4 主回归模型建议保留变量\n")
md.append("""
**控制变量：**
`log_fans`、`log_age`、`log_duration`、`tag_count`、`description_len`、
`domain` 哑变量、`pub_month` 哑变量

**标题特征：**
`title_char_len`（优先于 title_word_count）、`title_question_type`、
`title_exclamation_type`、`title_has_number`、`title_sentiment_intensity`、
`curiosity_marker_count`、`practical_value_word_count`

**封面特征：**
`image_brightness`、`image_colorfulness`、`visual_complexity`、
`cover_text_area_ratio`（优先于 cover_has_text）、`face_presence`

**双模态协同：**
`image_title_fit_c`（均值中心化）、`image_title_fit_c_sq`（中心化平方项）、
`title_cover_text_overlap`、`sentiment_consistency`、
`both_high_arousal`、`semantic_text_interaction`
""")

md.append("### 5.5 适合放入稳健性或替代模型的变量\n")
md.append("""
- `title_word_count`：`title_char_len` 的替代测量
- `cover_has_text`：`cover_text_area_ratio` 的二值替代
- `sentiment_conflict`：与 `sentiment_consistency` 互斥，用于对立假设模型
- `image_contrast`：与 `image_colorfulness` / `visual_complexity` 有一定重叠
- `log_view_per_day`、`recognition`、`participation`、`communication_effect`、
  `relative_view`：各维度因变量，分别作为主模型依变量进行敏感性分析
""")

md.append("\n---\n")
md.append("> *本报告由 Python 脚本自动生成，数据结论以实际输出为准。*")

with open(LOG_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
log.info(f"Markdown 报告：{LOG_MD}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Excel 输出
# ══════════════════════════════════════════════════════════════════════════════
log.info("写入 Excel…")

THIN  = Side(style="thin", color="000000")
BRD   = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HFILL = PatternFill("solid", fgColor="1F4E79")
HFONT = Font(bold=True, color="FFFFFF", size=10)
TFONT = Font(bold=True, size=11)
LFILL = PatternFill("solid", fgColor="DCE6F1")

def hdr(ws, row=1):
    for c in ws[row]:
        c.fill = HFILL; c.font = HFONT; c.border = BRD
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

def body(ws, start=2):
    for i, row in enumerate(ws.iter_rows(min_row=start, max_row=ws.max_row)):
        fl = LFILL if i % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")
        for c in row:
            c.border = BRD; c.fill = fl
            c.alignment = Alignment(horizontal="center", vertical="center")

def aw(ws, mn=8, mx=40):
    for col in ws.columns:
        lens = [len(str(c.value)) for c in col if c.value]
        if lens:
            ws.column_dimensions[get_column_letter(col[0].column)].width = \
                min(mx, max(mn, max(lens) * 1.15))

def write_corr_sheet(wb, sname, R, P, var_list, mlabel):
    vars_ = [v for v in var_list if v in R.index]
    n     = len(vars_)
    ws    = wb.create_sheet(sname)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n + 2)
    ws.cell(1, 1).value = (f"{mlabel}相关系数矩阵"
                           f"（N={N}；* p<0.05，** p<0.01，*** p<0.001）")
    ws.cell(1, 1).font      = TFONT
    ws.cell(1, 1).alignment = Alignment(horizontal="center")
    ws.cell(2, 1).value = "变量（英文）"
    ws.cell(2, 2).value = "中文含义"
    for j, v in enumerate(vars_):
        ws.cell(2, j + 3).value = LABEL.get(v, v)
    hdr(ws, 2)
    for i, vi in enumerate(vars_):
        r_ = i + 3
        ws.cell(r_, 1).value = vi
        ws.cell(r_, 2).value = LABEL.get(vi, vi)
        for j, vj in enumerate(vars_):
            rv = R.loc[vi, vj]
            pv = P.loc[vi, vj]
            if i == j:
                ws.cell(r_, j + 3).value = "—"
            else:
                star = sig_stars(pv)
                ws.cell(r_, j + 3).value = f"{rv:.3f}{star}"
                if abs(rv) >= 0.7:
                    fc = "C00000" if rv > 0 else "0070C0"
                    ws.cell(r_, j + 3).font = Font(bold=True, color=fc)
    body(ws, 3)
    aw(ws)

wb = Workbook()
wb.remove(wb.active)

write_corr_sheet(wb, "Pearson相关系数（全变量）",  PR, PP, AVAIL, "Pearson")
write_corr_sheet(wb, "Spearman相关系数（全变量）", SR, SP, AVAIL, "Spearman")
write_corr_sheet(wb, "Pearson相关系数（核心变量）", PR, PP, CORE_AVAIL, "Pearson（核心变量）")

# VIF Sheet
ws_v = wb.create_sheet("VIF多重共线性检验")
ws_v.merge_cells("A1:D1")
ws_v["A1"].value     = f"VIF多重共线性检验结果（连续变量已标准化；分类变量已哑变量化；N={N}）"
ws_v["A1"].font      = TFONT
ws_v["A1"].alignment = Alignment(horizontal="center")
for j, h in enumerate(["变量（英文）", "中文含义", "VIF", "判断"], 1):
    ws_v.cell(2, j).value = h
hdr(ws_v, 2)
VFILL = {
    "✓ 可接受":  PatternFill("solid", fgColor="E2EFDA"),
    "[WARN] 需关注":  PatternFill("solid", fgColor="FFF2CC"),
    "✗ 严重共线": PatternFill("solid", fgColor="FFD7D7"),
}
for i, row in df_vif_res.iterrows():
    r_ = i + 3
    for j, k in enumerate(["变量（英文）", "中文含义", "VIF", "判断"], 1):
        ws_v.cell(r_, j).value     = row[k]
        ws_v.cell(r_, j).border    = BRD
        ws_v.cell(r_, j).alignment = Alignment(horizontal="center", vertical="center")
        ws_v.cell(r_, j).fill      = VFILL.get(row["判断"], PatternFill("solid", fgColor="FFFFFF"))
aw(ws_v)

# 高度相关对汇总
ws_p = wb.create_sheet("高度相关变量对（|r|≥0.3）")
ws_p.merge_cells("A1:F1")
ws_p["A1"].value     = "变量对相关系数汇总（Pearson；|r|≥0.3）"
ws_p["A1"].font      = TFONT
ws_p["A1"].alignment = Alignment(horizontal="center")
hdrs_p = ["变量A（英文）", "含义A", "变量B（英文）", "含义B", "Pearson r", "类型"]
for j, h in enumerate(hdrs_p, 1):
    ws_p.cell(2, j).value = h
hdr(ws_p, 2)
for i, (a, b, r) in enumerate(pairs30):
    r_ = i + 3
    kind = ("DV-DV"   if (a in DV and b in DV) else
            "DV-解释" if (a in DV or b in DV) else "解释-解释")
    vals = [a, LABEL.get(a, a), b, LABEL.get(b, b), round(r, 4), kind]
    for j, v in enumerate(vals, 1):
        ws_p.cell(r_, j).value     = v
        ws_p.cell(r_, j).border    = BRD
        ws_p.cell(r_, j).alignment = Alignment(horizontal="center", vertical="center")
    if abs(r) >= 0.7:
        for j in range(1, 7):
            ws_p.cell(r_, j).font = Font(bold=True, color="C00000")
    elif abs(r) >= 0.5:
        for j in range(1, 7):
            ws_p.cell(r_, j).font = Font(bold=True, color="9C5700")
aw(ws_p)

# 变量说明表
ws_m = wb.create_sheet("变量说明表")
hdrs_m = ["变量类别", "变量（英文）", "中文含义", "类型", "是否进入主模型", "备注"]
for j, h in enumerate(hdrs_m, 1):
    ws_m.cell(1, j).value = h
hdr(ws_m, 1)
rows_m  = []
cat_map = {
    "DV":    ("因变量",     DV),
    "CTRL":  ("控制变量",   CTRL),
    "TITLE": ("标题特征",   TITLE),
    "COVER": ("封面特征",   COVER),
    "BIMOD": ("双模态特征", BIMOD),
}
for _, (cat_label, vlist) in cat_map.items():
    for v in vlist:
        if v not in AVAIL:
            continue
        dtype = "连续" if df_c[v].nunique() > 10 else "二值/有序"
        flag  = "✓"
        note  = ""
        if v == "title_word_count":          flag = "稳健性";   note = "与title_char_len高度相关"
        if v == "cover_has_text":            flag = "稳健性";   note = "与cover_text_area_ratio高度相关"
        if v == "sentiment_conflict":        flag = "补充模型"; note = "与sentiment_consistency互斥"
        if v == "image_title_fit_sq":        note = "建议用中心化平方项image_title_fit_c_sq替代"
        if v in ["cover_has_chart_or_formula", "cover_is_slide_or_screenshot"]:
            flag = "—"; note = "数据100%缺失，已从本步分析中排除"
        rows_m.append([cat_label, v, LABEL.get(v, v), dtype, flag, note])
for i, row in enumerate(rows_m):
    r_ = i + 2
    for j, val in enumerate(row, 1):
        ws_m.cell(r_, j).value     = val
        ws_m.cell(r_, j).border    = BRD
        ws_m.cell(r_, j).alignment = Alignment(horizontal="center", vertical="center")
aw(ws_m)

wb.save(XLSX_OUT)
log.info(f"Excel 保存：{XLSX_OUT}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. 终端摘要
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("【第七步完成】相关性与多重共线性检验")
print(f"  样本量（listwise）：N = {N}")
print(f"  可用变量数：{len(AVAIL)}（原始目标 {len(ALL_VARS)}）")
print(f"\n  高度相关对（|r|≥0.5）：{len(pairs50)} 对")
for a, b, r in pairs50[:8]:
    print(f"    {a} ↔ {b}: r={r:.3f}")
print(f"\n  VIF 严重共线（≥10）：{len(severe_vif)} 个")
print(f"  VIF 需关注（5–10）：{len(warn_vif)} 个")
print("\n  输出文件：")
print(f"    {XLSX_OUT}")
print(f"    {FIG_FULL}")
print(f"    {FIG_CORE}")
print(f"    {LOG_MD}")
print("=" * 62)
log.info("=== 第七步全部完成 ===")
