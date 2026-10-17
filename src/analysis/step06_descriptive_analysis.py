"""
=============================================================================
第六步：描述性统计与差异分析
研究题目：标题和封面双模态协同对社交媒体平台视频传播的影响——以B站知识类视频为例
作者：研究助理脚本（浙江大学论文格式）
=============================================================================

依赖包：
    pip install pandas numpy scipy matplotlib seaborn openpyxl

输入文件（均位于 outputs/ 目录，本脚本为兼容性也接受 uploads/ 路径）：
    - propagation_metrics.csv
    - text_features.csv
    - image_features.csv
    - ocr_features.csv
    - multimodal_features.csv

输出文件：
    - outputs/modeling_dataset.csv         合并后的建模数据集
    - outputs/descriptive_tables.xlsx      描述性统计表（多 Sheet）
    - outputs/difference_tests.xlsx        差异检验表（多 Sheet）
    - outputs/figures/domain_spread_bar.png
    - outputs/figures/multimodal_type_spread_bar.png
    - outputs/figures/high_low_image_title_fit_boxplot.png
    - outputs/figures/high_low_title_overlap_boxplot.png
    - outputs/figures/high_low_title_features_bar.png
    - outputs/logs/step6_quality_report.txt
=============================================================================
"""

# ── 0. 导入依赖 ────────────────────────────────────────────────────────────
import os
import sys
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合服务器/脚本环境
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import seaborn as sns
from openpyxl import load_workbook
from openpyxl.styles import (
    Font, Alignment, Border, Side, PatternFill, numbers
)
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")

# ── 1. 路径配置 ────────────────────────────────────────────────────────────
# 统一以运行时当前工作目录（仓库根目录）为基准，输入 outputs/*.csv、输出到 outputs/。
# 请在仓库根目录运行，或通过 run_pipeline.py 编排器调用（自动设置 cwd）。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
LOGS_DIR    = os.path.join(OUTPUT_DIR, "logs")

for d in [OUTPUT_DIR, UPLOAD_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "step6_quality_report.txt")

# ── 2. 日志系统 ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("第六步：描述性统计与差异分析  开始执行")
log.info(f"执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info("=" * 70)

# ── 3. matplotlib 中文字体配置 ────────────────────────────────────────────
def setup_chinese_font():
    """尝试加载系统中文字体，兼容 Linux / macOS / Windows。"""
    candidates = [
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            log.info(f"中文字体已设置为：{name}")
            return name
    # 兜底：仅在 Linux 环境尝试安装 fonts-wqy-microhei；Windows 不执行 apt-get。
    if os.name == "posix":
        os.system("apt-get install -y fonts-wqy-microhei -qq 2>/dev/null")
        font_manager._load_fontmanager(try_read_cache=False)
        available2 = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available2:
                rcParams["font.family"] = name
                rcParams["axes.unicode_minus"] = False
                log.info(f"中文字体（安装后）已设置为：{name}")
                return name
    log.warning("未找到合适的中文字体，图表中文可能乱码；将使用系统默认字体。")
    rcParams["axes.unicode_minus"] = False
    return None

FONT_NAME = setup_chinese_font()

# ── 4. 浙大论文绘图风格 ──────────────────────────────────────────────────
ZJU_BLUE   = "#003F88"
ZJU_GOLD   = "#C4973A"
ZJU_GRAY   = "#595959"
ZJU_LGRAY  = "#D9D9D9"
ZJU_RED    = "#9E1B32"
ZJU_GREEN  = "#3A7D44"

PALETTE_DOMAIN = [ZJU_BLUE, ZJU_GOLD, ZJU_RED, ZJU_GREEN]
PALETTE_GROUP  = {"高传播组": ZJU_BLUE, "低传播组": ZJU_GOLD}

rcParams.update({
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.facecolor": "white",
    "axes.spines.right": False,
    "axes.spines.top":   False,
    "axes.linewidth":    0.8,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "font.size":         10,
})

# ── 5. Excel 格式化辅助函数 ───────────────────────────────────────────────
THIN_BORDER = Border(
    left=Side(style="thin"),  right=Side(style="thin"),
    top=Side(style="thin"),   bottom=Side(style="thin"),
)
THICK_BOTTOM = Border(bottom=Side(style="medium"))
HEADER_FILL = PatternFill("solid", fgColor="003F88")   # 浙大蓝表头

def style_header_row(ws, row, max_col):
    for col in range(1, max_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.font      = Font(bold=True, color="FFFFFF", name="Times New Roman", size=10)
        cell.fill      = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = THIN_BORDER

def style_data_rows(ws, start_row, end_row, max_col):
    for r in range(start_row, end_row + 1):
        for c in range(1, max_col + 1):
            cell = ws.cell(row=r, column=c)
            cell.border    = THIN_BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.font      = Font(name="Times New Roman", size=10)

def set_col_widths(ws, widths):
    """widths: dict {col_letter: width} 或 list of widths"""
    if isinstance(widths, list):
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
    else:
        for letter, w in widths.items():
            ws.column_dimensions[letter].width = w

def write_note_row(ws, row, text, max_col):
    """在指定行写注释，跨列合并，斜体。"""
    ws.merge_cells(start_row=row, start_column=1,
                   end_row=row, end_column=max_col)
    cell = ws.cell(row=row, column=1)
    cell.value     = text
    cell.font      = Font(italic=True, size=9, name="Times New Roman")
    cell.alignment = Alignment(horizontal="left")

def sig_stars(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# ── 6. 读取输入文件 ───────────────────────────────────────────────────────
def try_read(name, prefer_outputs=True):
    """
    读取输入 CSV。

    修正点：
    1. 不再使用 /mnt/user-data/outputs 这种 Linux 沙盒路径；
    2. 优先读取脚本同级目录下的 outputs/；
    3. 兼容从其他工作目录运行脚本的情况；
    4. 报错时打印实际检查过的完整路径，便于排查。
    """
    filename = f"{name}.csv"

    script_output = os.path.join(SCRIPT_DIR, "outputs", filename)
    script_upload = os.path.join(SCRIPT_DIR, "uploads", filename)
    cwd_output = os.path.join(os.getcwd(), "outputs", filename)
    cwd_upload = os.path.join(os.getcwd(), "uploads", filename)

    if prefer_outputs:
        paths = [script_output, cwd_output, script_upload, cwd_upload]
    else:
        paths = [script_upload, cwd_upload, script_output, cwd_output]

    # 去重，但保留顺序
    paths = list(dict.fromkeys(os.path.abspath(p) for p in paths))

    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            log.info(f"  已读取 {p}  →  {df.shape[0]} 行 × {df.shape[1]} 列")
            return df

    log.error(f"  !! 文件 {filename} 未找到。已检查以下路径：")
    for p in paths:
        log.error(f"     - {p}")
    raise FileNotFoundError(f"{filename} not found. Checked: {paths}")

log.info("\n[6-1] 读取输入文件 ...")
df_prop  = try_read("propagation_metrics")
df_text  = try_read("text_features")
df_img   = try_read("image_features")
df_ocr   = try_read("ocr_features")
df_multi = try_read("multimodal_features")

# ── 7. 合并数据集 ────────────────────────────────────────────────────────
log.info("\n[6-2] 以 bvid 为主键合并数据 ...")

# 确保 bvid 为字符串
for df in [df_prop, df_text, df_img, df_ocr, df_multi]:
    df["bvid"] = df["bvid"].astype(str).str.strip()

# 逐步左连接（以 propagation_metrics 为基准）
df = df_prop.copy()
for src, name in [(df_text, "text_features"),
                  (df_img,  "image_features"),
                  (df_ocr,  "ocr_features"),
                  (df_multi,"multimodal_features")]:
    before = len(df)
    overlap_cols = [c for c in src.columns if c != "bvid" and c in df.columns]
    if overlap_cols:
        log.info(f"  合并 {name}：跳过重复列 {overlap_cols}")
        src = src.drop(columns=overlap_cols)
    df = df.merge(src, on="bvid", how="left")
    after = len(df)
    if before != after:
        log.warning(f"  合并 {name} 后行数变化：{before} → {after}（请检查是否存在重复 bvid）")

log.info(f"  合并后数据集：{df.shape[0]} 行 × {df.shape[1]} 列")

# 检查缺失列并补 NaN（兼容不同步骤生成的特征文件）
EXPECTED_COLS = [
    "cover_has_chart_or_formula", "cover_is_slide_or_screenshot", "cover_acgn_style"
]
for col in EXPECTED_COLS:
    if col not in df.columns:
        df[col] = np.nan
        log.warning(f"  列 {col} 不存在于任何输入文件，已填充 NaN，请在后续步骤补全。")

# domain 标准化
DOMAIN_MAP = {
    "科学科普": "科学科普", "科普": "科学科普",
    "社科人文": "社科人文", "人文": "社科人文",
    "财经商业": "财经商业", "财经": "财经商业",
    "职场教育": "职场教育", "教育": "职场教育",
}
df["domain"] = df["domain"].map(lambda x: DOMAIN_MAP.get(str(x).strip(), str(x).strip()))
unexpected_domain = df[~df["domain"].isin(DOMAIN_MAP.values())]["domain"].unique()
if len(unexpected_domain) > 0:
    log.warning(f"  domain 存在未识别值：{unexpected_domain}")

# 缺失值检查
null_summary = df.isnull().sum()
null_cols = null_summary[null_summary > 0]
if len(null_cols) > 0:
    log.info(f"  存在缺失值的列（共 {len(null_cols)} 列）：")
    for col, cnt in null_cols.items():
        log.info(f"    {col}: {cnt} 缺失 ({cnt/len(df)*100:.1f}%)")
else:
    log.info("  数据集无缺失值。")

# 保存建模数据集
out_model = os.path.join(OUTPUT_DIR, "modeling_dataset.csv")
df.to_csv(out_model, index=False, encoding="utf-8-sig")
log.info(f"\n  已保存建模数据集 → {out_model}")

# ── 8. 辅助函数：连续变量描述统计 ────────────────────────────────────────
def describe_continuous(data, cols, fmt=".3f"):
    """返回均值/中位数/标准差/最小/最大/N 的 DataFrame。"""
    rows = []
    for c in cols:
        if c not in data.columns:
            rows.append({"变量": c, "N": "—", "均值": "—", "中位数": "—",
                         "标准差": "—", "最小值": "—", "最大值": "—"})
            continue
        s = data[c].dropna()
        rows.append({
            "变量":   c,
            "N":      len(s),
            "均值":   f"{s.mean():{fmt}}",
            "中位数": f"{s.median():{fmt}}",
            "标准差": f"{s.std():{fmt}}",
            "最小值": f"{s.min():{fmt}}",
            "最大值": f"{s.max():{fmt}}",
        })
    return pd.DataFrame(rows)

def describe_categorical(data, col):
    """返回分类变量频数和占比表。"""
    vc = data[col].value_counts(dropna=False)
    pct = (vc / len(data) * 100).round(2)
    return pd.DataFrame({"类别": vc.index, "频数": vc.values,
                          "占比(%)": pct.values})

# ── 9. 表1：样本基本情况 ─────────────────────────────────────────────────
log.info("\n[6-3] 生成表1：样本基本情况 ...")

sample_info = {
    "样本总数":  len(df),
    "UP主数量": df["uid"].nunique(),
}
log.info(f"  样本总数：{sample_info['样本总数']}，UP主数量：{sample_info['UP主数量']}")

# domain 分布
domain_dist = df["domain"].value_counts().reset_index()
domain_dist.columns = ["领域", "视频数"]
domain_dist["占比(%)"] = (domain_dist["视频数"] / len(df) * 100).round(2)

# level 分布
level_dist = df["level"].value_counts().sort_index().reset_index()
level_dist.columns = ["层级", "视频数"]
level_dist["占比(%)"] = (level_dist["视频数"] / len(df) * 100).round(2)

# 核心连续变量描述
CORE_CONT_VARS = [
    ("view",             "播放量",         ".0f"),
    ("like",             "点赞数",         ".0f"),
    ("coin",             "投币数",         ".0f"),
    ("favorite",         "收藏数",         ".0f"),
    ("share",            "转发数",         ".0f"),
    ("danmaku",          "弹幕数",         ".0f"),
    ("reply",            "评论数",         ".0f"),
    ("duration",         "视频时长(秒)",   ".0f"),
    ("fans",             "粉丝数",         ".0f"),
    ("video_age_days",   "发布天数",       ".1f"),
    ("log_view",         "log播放量",      ".3f"),
    ("log_view_per_day", "log日均播放量",  ".3f"),
    ("communication_effect", "传播效果综合指标", ".3f"),
]

rows_cont = []
for varname, alias, fmt in CORE_CONT_VARS:
    if varname not in df.columns:
        log.warning(f"  列 {varname} 不存在，跳过。")
        continue
    s = df[varname].dropna()
    rows_cont.append({
        "变量名(英文)":    varname,
        "中文别名":       alias,
        "N":             f"{len(s):,}",
        "均值":          f"{s.mean():{fmt}}",
        "中位数":        f"{s.median():{fmt}}",
        "标准差":        f"{s.std():{fmt}}",
        "最小值":        f"{s.min():{fmt}}",
        "最大值":        f"{s.max():{fmt}}",
    })

table1_cont = pd.DataFrame(rows_cont)

# ── 10. 表2：不同知识领域传播效果 ─────────────────────────────────────────
log.info("\n[6-4] 生成表2：不同知识领域传播效果比较 ...")

DOMAIN_ORDER = ["科学科普", "社科人文", "财经商业", "职场教育"]
agg_dict = {}
domain_vars = ["view", "log_view", "log_view_per_day",
               "recognition", "participation", "communication_effect"]
for v in domain_vars:
    if v in df.columns:
        agg_dict[v] = "mean"

table2 = df.groupby("domain").agg(agg_dict).reset_index()
table2.columns = ["领域"] + [f"平均{c}" for c in agg_dict.keys()]
table2["样本量"] = df.groupby("domain").size().values
# 按预设顺序排列
table2["领域"] = pd.Categorical(table2["领域"], categories=DOMAIN_ORDER, ordered=True)
table2 = table2.sort_values("领域").reset_index(drop=True)
# 格式化
for col in table2.columns[1:]:
    if col == "样本量":
        continue
    try:
        table2[col] = table2[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    except Exception:
        pass

# ── 11. 表3：标题文本特征描述 ──────────────────────────────────────────────
log.info("\n[6-5] 生成表3：标题文本特征描述 ...")

TEXT_VARS_CONT = [
    ("title_char_len",           "标题字符数",     ".2f"),
    ("title_word_count",         "标题词数",       ".2f"),
    ("title_sentiment_intensity","情感强度",       ".3f"),
    ("curiosity_marker_count",   "好奇心标记词数", ".3f"),
    ("practical_value_word_count","实用价值词数",  ".3f"),
]
TEXT_VARS_CAT = [
    ("title_question_type",      "疑问句类型",     "分类"),
    ("title_exclamation_type",   "感叹句类型",     "分类"),
    ("title_has_number",         "含数字",         "二值"),
    ("title_has_practical_value","含实用价值词",   "二值"),
]

rows_t3 = []
for varname, alias, fmt in TEXT_VARS_CONT:
    if varname not in df.columns:
        rows_t3.append({"变量名": varname, "中文别名": alias, "类型": "连续",
                        "均值/占比": "—", "中位数": "—", "标准差": "—", "取值范围": "—"})
        continue
    s = df[varname].dropna()
    rows_t3.append({
        "变量名":   varname,
        "中文别名": alias,
        "类型":     "连续",
        "均值/占比": f"{s.mean():{fmt}}",
        "中位数":   f"{s.median():{fmt}}",
        "标准差":   f"{s.std():{fmt}}",
        "取值范围": f"[{s.min():{fmt}}, {s.max():{fmt}}]",
    })

for varname, alias, vtype in TEXT_VARS_CAT:
    if varname not in df.columns:
        rows_t3.append({"变量名": varname, "中文别名": alias, "类型": vtype,
                        "均值/占比": "—", "中位数": "—", "标准差": "—", "取值范围": "—"})
        continue
    s = df[varname].dropna()
    if vtype == "二值":
        prop = s.mean()
        rows_t3.append({
            "变量名":   varname,
            "中文别名": alias,
            "类型":     vtype,
            "均值/占比": f"{prop:.1%}",
            "中位数":   "—",
            "标准差":   "—",
            "取值范围": "0/1",
        })
    else:  # 分类
        vc = s.value_counts()
        desc = "; ".join([f"{k}:{v}({v/len(s):.1%})" for k, v in vc.items()])
        rows_t3.append({
            "变量名":   varname,
            "中文别名": alias,
            "类型":     vtype,
            "均值/占比": desc[:60],
            "中位数":   "—",
            "标准差":   "—",
            "取值范围": "—",
        })

table3 = pd.DataFrame(rows_t3)

# ── 12. 表4：封面图像特征描述 ──────────────────────────────────────────────
log.info("\n[6-6] 生成表4：封面图像特征描述 ...")

IMG_VARS_CONT = [
    ("image_brightness",     "图像亮度",     ".2f"),
    ("image_contrast",       "图像对比度",   ".2f"),
    ("image_colorfulness",   "色彩丰富度",   ".2f"),
    ("image_saturation",     "图像饱和度",   ".2f"),
    ("visual_complexity",    "视觉复杂度",   ".3f"),
    ("cover_text_area_ratio","文字面积比",   ".3f"),
]
IMG_VARS_CAT = [
    ("cover_has_text",              "封面含文字",     "二值"),
    ("human_presence",              "有人物",         "二值"),
    ("face_presence",               "有人脸",         "二值"),
    ("cover_has_chart_or_formula",  "含图表/公式",    "二值"),
    ("cover_is_slide_or_screenshot","截图/幻灯片",    "二值"),
    ("cover_acgn_style",            "ACGN风格",       "二值"),
]

rows_t4 = []
for varname, alias, fmt in IMG_VARS_CONT:
    if varname not in df.columns or df[varname].isna().all():
        rows_t4.append({"变量名": varname, "中文别名": alias, "类型": "连续",
                        "均值/占比": "—", "中位数": "—", "标准差": "—", "取值范围": "—"})
        continue
    s = df[varname].dropna()
    rows_t4.append({
        "变量名":   varname,
        "中文别名": alias,
        "类型":     "连续",
        "均值/占比": f"{s.mean():{fmt}}",
        "中位数":   f"{s.median():{fmt}}",
        "标准差":   f"{s.std():{fmt}}",
        "取值范围": f"[{s.min():{fmt}}, {s.max():{fmt}}]",
    })

for varname, alias, vtype in IMG_VARS_CAT:
    if varname not in df.columns or df[varname].isna().all():
        rows_t4.append({"变量名": varname, "中文别名": alias, "类型": vtype,
                        "均值/占比": "数据待补全", "中位数": "—",
                        "标准差": "—", "取值范围": "0/1"})
        log.warning(f"  表4：列 {varname} 数据缺失，标注为'数据待补全'")
        continue
    s = df[varname].dropna()
    prop = s.mean()
    rows_t4.append({
        "变量名":   varname,
        "中文别名": alias,
        "类型":     vtype,
        "均值/占比": f"{prop:.1%}",
        "中位数":   "—",
        "标准差":   "—",
        "取值范围": "0/1",
    })

table4 = pd.DataFrame(rows_t4)

# ── 13. 表5：双模态协同特征描述 ───────────────────────────────────────────
log.info("\n[6-7] 生成表5：双模态协同特征描述 ...")

MULTI_VARS_CONT = [
    ("image_title_fit",          "语义一致性(CLIP)",   ".3f"),
    ("title_cover_text_overlap", "文字重叠度",         ".3f"),
    ("sentiment_intensity_gap",  "情感强度差",         ".3f"),
]
MULTI_VARS_CAT = [
    ("sentiment_consistency",  "情感一致",       "二值"),
    ("sentiment_conflict",     "情感冲突",       "二值"),
    ("both_high_arousal",      "双高唤醒",       "二值"),
    ("complementarity_type",   "互补类型",       "分类"),
]

rows_t5 = []
for varname, alias, fmt in MULTI_VARS_CONT:
    if varname not in df.columns or df[varname].isna().all():
        rows_t5.append({"变量名": varname, "中文别名": alias, "类型": "连续",
                        "均值/占比": "—", "中位数": "—", "标准差": "—", "取值范围": "—"})
        continue
    s = df[varname].dropna()
    rows_t5.append({
        "变量名":   varname,
        "中文别名": alias,
        "类型":     "连续",
        "均值/占比": f"{s.mean():{fmt}}",
        "中位数":   f"{s.median():{fmt}}",
        "标准差":   f"{s.std():{fmt}}",
        "取值范围": f"[{s.min():{fmt}}, {s.max():{fmt}}]",
    })

for varname, alias, vtype in MULTI_VARS_CAT:
    if varname not in df.columns or df[varname].isna().all():
        rows_t5.append({"变量名": varname, "中文别名": alias, "类型": vtype,
                        "均值/占比": "—", "中位数": "—", "标准差": "—", "取值范围": "—"})
        continue
    s = df[varname].dropna()
    if vtype == "二值":
        prop = s.mean()
        rows_t5.append({
            "变量名":   varname,
            "中文别名": alias,
            "类型":     vtype,
            "均值/占比": f"{prop:.1%}",
            "中位数":   "—",
            "标准差":   "—",
            "取值范围": "0/1",
        })
    else:
        vc = s.value_counts()
        desc = "; ".join([f"{k}:{v}({v/len(s):.1%})" for k, v in vc.items()])
        rows_t5.append({
            "变量名":   varname,
            "中文别名": alias,
            "类型":     vtype,
            "均值/占比": desc,
            "中位数":   "—",
            "标准差":   "—",
            "取值范围": "多类",
        })

table5 = pd.DataFrame(rows_t5)

# ── 14. 分组构造 ──────────────────────────────────────────────────────────
log.info("\n[6-8] 构造高/低传播组 ...")

TARGET = "log_view_per_day"
med = df[TARGET].median()
q25 = df[TARGET].quantile(0.25)
q75 = df[TARGET].quantile(0.75)
log.info(f"  {TARGET}: 中位数={med:.4f}, Q25={q25:.4f}, Q75={q75:.4f}")

# median_group
df["median_group"] = np.where(df[TARGET] >= med, "高传播组", "低传播组")
log.info(f"  median_group: 高={( df['median_group']=='高传播组').sum()}, 低={(df['median_group']=='低传播组').sum()}")

# q_group（中间样本排除）
df["q_group"] = np.where(df[TARGET] >= q75, "高传播组",
               np.where(df[TARGET] <= q25, "低传播组", "中间"))
n_high_q = (df["q_group"] == "高传播组").sum()
n_low_q  = (df["q_group"] == "低传播组").sum()
n_mid_q  = (df["q_group"] == "中间").sum()
log.info(f"  q_group: 高={n_high_q}, 中间={n_mid_q}(排除), 低={n_low_q}")

# ── 15. 差异检验函数 ──────────────────────────────────────────────────────
def diff_test_continuous(df_sub, col, group_col="median_group",
                          hi_label="高传播组", lo_label="低传播组"):
    """对连续变量同时输出 t 检验和 Mann-Whitney U 检验。"""
    hi = df_sub[df_sub[group_col] == hi_label][col].dropna()
    lo = df_sub[df_sub[group_col] == lo_label][col].dropna()
    if len(hi) < 5 or len(lo) < 5:
        return None
    mean_hi, mean_lo = hi.mean(), lo.mean()
    # t 检验
    t_stat, t_p = stats.ttest_ind(hi, lo, equal_var=False)
    # Mann-Whitney U
    mw_stat, mw_p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
    return {
        "高传播组均值": mean_hi,
        "低传播组均值": mean_lo,
        "差值(高-低)":  mean_hi - mean_lo,
        "t统计量":      t_stat,
        "t_p值":        t_p,
        "t显著性":      sig_stars(t_p),
        "MWU统计量":    mw_stat,
        "MWU_p值":      mw_p,
        "MWU显著性":    sig_stars(mw_p),
    }

def diff_test_categorical(df_sub, col, group_col="median_group",
                           hi_label="高传播组", lo_label="低传播组"):
    """对分类/二值变量输出卡方检验。"""
    sub = df_sub[[group_col, col]].dropna()
    ct = pd.crosstab(sub[group_col], sub[col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    hi = sub[sub[group_col] == hi_label][col]
    lo = sub[sub[group_col] == lo_label][col]
    # 对二值变量计算占比
    if set(sub[col].unique()).issubset({0, 1, True, False, 0.0, 1.0}):
        prop_hi = hi.mean()
        prop_lo = lo.mean()
        desc_hi = f"{prop_hi:.1%}"
        desc_lo = f"{prop_lo:.1%}"
        diff    = f"{prop_hi - prop_lo:+.1%}"
    else:
        # 多分类：列出最多两个类别的占比
        vc_hi = hi.value_counts(normalize=True).to_dict()
        vc_lo = lo.value_counts(normalize=True).to_dict()
        desc_hi = "; ".join([f"{k}:{v:.1%}" for k, v in list(vc_hi.items())[:3]])
        desc_lo = "; ".join([f"{k}:{v:.1%}" for k, v in list(vc_lo.items())[:3]])
        diff = "—"
    return {
        "高传播组":    desc_hi,
        "低传播组":    desc_lo,
        "差值":        diff,
        "卡方统计量":  chi2,
        "自由度":      dof,
        "p值":         p,
        "显著性":      sig_stars(p),
    }

# ── 16. 差异检验变量清单 ──────────────────────────────────────────────────
TEST_VARS_CONT = [
    # 标题特征
    ("title_char_len",            "标题字符数",          "标题特征"),
    ("title_word_count",          "标题词数",            "标题特征"),
    ("title_sentiment_intensity", "情感强度",            "标题特征"),
    ("curiosity_marker_count",    "好奇心标记词数",      "标题特征"),
    ("practical_value_word_count","实用价值词数",        "标题特征"),
    # 封面特征
    ("image_brightness",          "图像亮度",            "封面特征"),
    ("image_contrast",            "图像对比度",          "封面特征"),
    ("image_colorfulness",        "色彩丰富度",          "封面特征"),
    ("image_saturation",          "图像饱和度",          "封面特征"),
    ("visual_complexity",         "视觉复杂度",          "封面特征"),
    ("cover_text_area_ratio",     "文字面积比",          "封面特征"),
    # 双模态协同
    ("image_title_fit",           "语义一致性",          "双模态协同"),
    ("title_cover_text_overlap",  "文字重叠度",          "双模态协同"),
    ("sentiment_intensity_gap",   "情感强度差",          "双模态协同"),
    # 控制变量
    ("log_fans",                  "log粉丝数",           "控制变量"),
    ("log_duration",              "log视频时长",         "控制变量"),
    ("video_age_days",            "发布天数",            "控制变量"),
    ("tag_count",                 "标签数",              "控制变量"),
    ("description_len",           "简介长度",            "控制变量"),
]

TEST_VARS_CAT = [
    # 标题特征
    ("title_question_type",       "疑问句类型",          "标题特征"),
    ("title_exclamation_type",    "感叹句类型",          "标题特征"),
    ("title_has_number",          "含数字",              "标题特征"),
    ("title_has_practical_value", "含实用价值词",        "标题特征"),
    # 封面特征
    ("cover_has_text",            "封面含文字",          "封面特征"),
    ("human_presence",            "有人物",              "封面特征"),
    ("face_presence",             "有人脸",              "封面特征"),
    ("cover_has_chart_or_formula","含图表/公式",         "封面特征"),
    ("cover_is_slide_or_screenshot","截图/幻灯片",       "封面特征"),
    ("cover_acgn_style",          "ACGN风格",            "封面特征"),
    # 双模态协同
    ("sentiment_consistency",     "情感一致",            "双模态协同"),
    ("sentiment_conflict",        "情感冲突",            "双模态协同"),
    ("both_high_arousal",         "双高唤醒",            "双模态协同"),
    ("complementarity_type",      "互补类型",            "双模态协同"),
    # 控制变量
    ("domain",                    "知识领域",            "控制变量"),
    ("level",                     "UP主层级",            "控制变量"),
]

# ── 17. 执行差异检验（median & q 两种分组）──────────────────────────────
log.info("\n[6-9] 执行差异检验 ...")

def run_diff_tests(df_input, group_col):
    """对所有变量执行差异检验，返回结果 DataFrame。"""
    rows = []
    df_q = df_input[df_input[group_col] != "中间"].copy()  # q_group 过滤中间

    for varname, alias, category in TEST_VARS_CONT:
        if varname not in df_q.columns or df_q[varname].isna().all():
            log.warning(f"  [差异检验] 跳过连续变量 {varname}（列不存在或全为 NaN）")
            rows.append({
                "类别": category, "变量名": varname, "中文别名": alias,
                "类型": "连续",
                "高传播组均值": "—", "低传播组均值": "—", "差值(高-低)": "—",
                "t统计量": "—", "t_p值": "—", "t显著性": "—",
                "MWU统计量": "—", "MWU_p值": "—", "MWU显著性": "—",
                "卡方统计量": "—", "自由度": "—", "p值": "—", "显著性": "—",
            })
            continue
        res = diff_test_continuous(df_q, varname, group_col=group_col)
        if res is None:
            continue
        rows.append({
            "类别": category, "变量名": varname, "中文别名": alias,
            "类型": "连续",
            "高传播组均值": f"{res['高传播组均值']:.3f}",
            "低传播组均值": f"{res['低传播组均值']:.3f}",
            "差值(高-低)":  f"{res['差值(高-低)']:+.3f}",
            "t统计量":      f"{res['t统计量']:.3f}",
            "t_p值":        f"{res['t_p值']:.4f}",
            "t显著性":      res["t显著性"],
            "MWU统计量":    f"{res['MWU统计量']:.1f}",
            "MWU_p值":      f"{res['MWU_p值']:.4f}",
            "MWU显著性":    res["MWU显著性"],
            "卡方统计量": "—", "自由度": "—", "p值": "—", "显著性": "—",
        })

    for varname, alias, category in TEST_VARS_CAT:
        if varname not in df_q.columns or df_q[varname].isna().all():
            log.warning(f"  [差异检验] 跳过分类变量 {varname}（列不存在或全为 NaN）")
            rows.append({
                "类别": category, "变量名": varname, "中文别名": alias,
                "类型": "分类",
                "高传播组均值": "—", "低传播组均值": "—", "差值(高-低)": "—",
                "t统计量": "—", "t_p值": "—", "t显著性": "—",
                "MWU统计量": "—", "MWU_p值": "—", "MWU显著性": "—",
                "卡方统计量": "—", "自由度": "—", "p值": "—", "显著性": "—",
            })
            continue
        res = diff_test_categorical(df_q, varname, group_col=group_col)
        if res is None:
            continue
        rows.append({
            "类别": category, "变量名": varname, "中文别名": alias,
            "类型": "分类",
            "高传播组均值": res["高传播组"],
            "低传播组均值": res["低传播组"],
            "差值(高-低)":  res["差值"],
            "t统计量": "—", "t_p值": "—", "t显著性": "—",
            "MWU统计量": "—", "MWU_p值": "—", "MWU显著性": "—",
            "卡方统计量": f"{res['卡方统计量']:.3f}",
            "自由度":     f"{res['自由度']}",
            "p值":        f"{res['p值']:.4f}",
            "显著性":     res["显著性"],
        })
    return pd.DataFrame(rows)

diff_median = run_diff_tests(df, "median_group")
diff_q      = run_diff_tests(df, "q_group")
log.info(f"  median_group 差异检验完成：{len(diff_median)} 行")
log.info(f"  q_group      差异检验完成：{len(diff_q)} 行")

# ── 18. 保存 Excel ────────────────────────────────────────────────────────
log.info("\n[6-10] 写入 Excel 文件 ...")

# ─ 18a. descriptive_tables.xlsx ─
desc_path = os.path.join(OUTPUT_DIR, "descriptive_tables.xlsx")
with pd.ExcelWriter(desc_path, engine="openpyxl") as writer:

    # ── Sheet1：样本基本情况 ──
    wb_sheet = "表1_样本基本情况"
    row_cursor = 1

    # 先写到临时 sheet
    blank_df = pd.DataFrame()
    blank_df.to_excel(writer, sheet_name=wb_sheet, index=False)
    ws1 = writer.sheets[wb_sheet]

    def w(ws, r, c, val, **kw):
        cell = ws.cell(row=r, column=c, value=val)
        return cell

    # 标题
    ws1.merge_cells("A1:H1")
    c = ws1["A1"]
    c.value = "表1  样本基本情况"
    c.font  = Font(bold=True, size=12, name="Times New Roman")
    c.alignment = Alignment(horizontal="center")

    # 基本数量
    ws1["A3"] = "样本总数"
    ws1["B3"] = sample_info["样本总数"]
    ws1["A4"] = "涉及UP主数"
    ws1["B4"] = sample_info["UP主数量"]
    for r in [3, 4]:
        for c_idx in range(1, 3):
            ws1.cell(r, c_idx).font = Font(name="Times New Roman", size=10)
            ws1.cell(r, c_idx).border = THIN_BORDER

    # domain 分布
    ws1["A6"] = "知识领域分布"
    ws1["A6"].font = Font(bold=True, name="Times New Roman", size=10)
    header = ["领域", "视频数", "占比(%)"]
    for ci, h in enumerate(header, 1):
        ws1.cell(7, ci).value = h
    style_header_row(ws1, 7, len(header))
    for ri, row_data in enumerate(domain_dist.itertuples(index=False), 8):
        for ci, val in enumerate(row_data, 1):
            ws1.cell(ri, ci).value = val
    style_data_rows(ws1, 8, 7 + len(domain_dist), len(header))

    # level 分布
    lv_start = 8 + len(domain_dist) + 2
    ws1.cell(lv_start, 1).value = "UP主层级分布"
    ws1.cell(lv_start, 1).font = Font(bold=True, name="Times New Roman", size=10)
    lv_header_row = lv_start + 1
    for ci, h in enumerate(["层级", "视频数", "占比(%)"], 1):
        ws1.cell(lv_header_row, ci).value = h
    style_header_row(ws1, lv_header_row, 3)
    for ri, row_data in enumerate(level_dist.itertuples(index=False), lv_header_row + 1):
        for ci, val in enumerate(row_data, 1):
            ws1.cell(ri, ci).value = val
    style_data_rows(ws1, lv_header_row + 1, lv_header_row + len(level_dist), 3)

    # 连续变量描述
    cont_start = lv_header_row + len(level_dist) + 2
    ws1.cell(cont_start, 1).value = "核心变量描述统计"
    ws1.cell(cont_start, 1).font = Font(bold=True, name="Times New Roman", size=10)
    cont_header_row = cont_start + 1
    cont_headers = ["变量名(英文)", "中文别名", "N", "均值", "中位数", "标准差", "最小值", "最大值"]
    for ci, h in enumerate(cont_headers, 1):
        ws1.cell(cont_header_row, ci).value = h
    style_header_row(ws1, cont_header_row, len(cont_headers))
    for ri, row_data in enumerate(table1_cont.itertuples(index=False), cont_header_row + 1):
        for ci, val in enumerate(row_data, 1):
            ws1.cell(ri, ci).value = val
    style_data_rows(ws1, cont_header_row + 1,
                    cont_header_row + len(table1_cont), len(cont_headers))
    note_row = cont_header_row + len(table1_cont) + 1
    write_note_row(ws1, note_row,
                   "注：播放量、点赞数等原始计数变量分布严重右偏，实证分析中均取自然对数。",
                   len(cont_headers))
    set_col_widths(ws1, [18, 16, 8, 12, 12, 12, 12, 12])

    # ── Sheet2：领域传播效果 ──
    table2.to_excel(writer, sheet_name="表2_领域传播效果", index=False)
    ws2 = writer.sheets["表2_领域传播效果"]
    style_header_row(ws2, 1, len(table2.columns))
    style_data_rows(ws2, 2, 1 + len(table2), len(table2.columns))
    note_r = 2 + len(table2)
    ws2.merge_cells(start_row=note_r, start_column=1, end_row=note_r, end_column=len(table2.columns))
    ws2.cell(note_r, 1).value = "注：recognition=认可度指数，participation=参与度指数，communication_effect=综合传播效果指数（Z分标准化合成）。"
    ws2.cell(note_r, 1).font = Font(italic=True, size=9, name="Times New Roman")
    set_col_widths(ws2, [12, 12, 12, 12, 12, 16, 8])

    # ── Sheet3：标题特征 ──
    table3.to_excel(writer, sheet_name="表3_标题文本特征", index=False)
    ws3 = writer.sheets["表3_标题文本特征"]
    style_header_row(ws3, 1, len(table3.columns))
    style_data_rows(ws3, 2, 1 + len(table3), len(table3.columns))
    set_col_widths(ws3, [22, 16, 8, 30, 12, 12, 18])

    # ── Sheet4：封面特征 ──
    table4.to_excel(writer, sheet_name="表4_封面图像特征", index=False)
    ws4 = writer.sheets["表4_封面图像特征"]
    style_header_row(ws4, 1, len(table4.columns))
    style_data_rows(ws4, 2, 1 + len(table4), len(table4.columns))
    note_r4 = 2 + len(table4)
    ws4.merge_cells(start_row=note_r4, start_column=1, end_row=note_r4, end_column=len(table4.columns))
    ws4.cell(note_r4, 1).value = "注：cover_has_chart_or_formula、cover_is_slide_or_screenshot、cover_acgn_style 需在图像分类步骤（第四步）补全，当前标注为'数据待补全'。"
    ws4.cell(note_r4, 1).font = Font(italic=True, size=9, name="Times New Roman")
    set_col_widths(ws4, [26, 14, 8, 20, 12, 12, 14])

    # ── Sheet5：双模态特征 ──
    table5.to_excel(writer, sheet_name="表5_双模态协同特征", index=False)
    ws5 = writer.sheets["表5_双模态协同特征"]
    style_header_row(ws5, 1, len(table5.columns))
    style_data_rows(ws5, 2, 1 + len(table5), len(table5.columns))
    note_r5 = 2 + len(table5)
    ws5.merge_cells(start_row=note_r5, start_column=1, end_row=note_r5, end_column=len(table5.columns))
    ws5.cell(note_r5, 1).value = (
        "注：image_title_fit 为 CLIP 模型计算的标题—封面语义相似度（代理方法）；"
        "title_cover_text_overlap 为 TF-IDF 余弦相似度与 Jaccard 相似度的综合重叠度；"
        "complementarity_type 为基于规则的双模态互补类型分类（6类）。"
    )
    ws5.cell(note_r5, 1).font = Font(italic=True, size=9, name="Times New Roman")
    set_col_widths(ws5, [24, 16, 8, 50, 12, 12, 14])

log.info(f"  已保存 → {desc_path}")

# ─ 18b. difference_tests.xlsx ─
diff_path = os.path.join(OUTPUT_DIR, "difference_tests.xlsx")
with pd.ExcelWriter(diff_path, engine="openpyxl") as writer:
    for sheet_name, diff_df, note in [
        ("中位数分组_差异检验", diff_median,
         "注：median_group 按 log_view_per_day 中位数分为高传播组（≥中位数）和低传播组（<中位数）。"
         "连续变量同时报告 Welch t 检验和 Mann-Whitney U 检验；分类变量报告卡方检验。"
         "显著性水平：***p<0.001，**p<0.01，*p<0.05。"),
        ("四分位分组_差异检验", diff_q,
         "注：q_group 按 log_view_per_day 四分位数分组，前25%为高传播组，后25%为低传播组，中间50%不参与检验。"
         "显著性水平：***p<0.001，**p<0.01，*p<0.05。"),
    ]:
        diff_df.to_excel(writer, sheet_name=sheet_name, index=False)
        ws = writer.sheets[sheet_name]
        style_header_row(ws, 1, len(diff_df.columns))
        style_data_rows(ws, 2, 1 + len(diff_df), len(diff_df.columns))
        # 显著性星号列加粗红色
        sig_cols = ["t显著性", "MWU显著性", "显著性"]
        for sig_col in sig_cols:
            if sig_col in diff_df.columns:
                col_idx = list(diff_df.columns).index(sig_col) + 1
                for ri in range(2, 2 + len(diff_df)):
                    cell = ws.cell(ri, col_idx)
                    if cell.value and cell.value.strip("*") == "":
                        cell.font = Font(bold=True, color="9E1B32",
                                         name="Times New Roman", size=10)
        # 按类别隔行着色（浅蓝 / 白）
        cat_fill = PatternFill("solid", fgColor="E8F0F7")
        prev_cat = None
        fill_toggle = False
        cat_col_idx = list(diff_df.columns).index("类别") + 1
        for ri in range(2, 2 + len(diff_df)):
            cat_val = ws.cell(ri, cat_col_idx).value
            if cat_val != prev_cat:
                fill_toggle = not fill_toggle
                prev_cat = cat_val
            if fill_toggle:
                for ci in range(1, len(diff_df.columns) + 1):
                    ws.cell(ri, ci).fill = cat_fill
        note_r = 2 + len(diff_df)
        ws.merge_cells(start_row=note_r, start_column=1,
                       end_row=note_r, end_column=len(diff_df.columns))
        ws.cell(note_r, 1).value = note
        ws.cell(note_r, 1).font = Font(italic=True, size=9, name="Times New Roman")
        # 列宽
        widths = [10, 22, 14, 6, 12, 12, 12, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8]
        for i, w in enumerate(widths[:len(diff_df.columns)], 1):
            ws.column_dimensions[get_column_letter(i)].width = w

log.info(f"  已保存 → {diff_path}")

# ── 19. 图表绘制 ──────────────────────────────────────────────────────────
log.info("\n[6-11] 绘制图表 ...")

# ─ 图1：不同 domain 平均 log_view_per_day 柱状图 ─
fig, ax = plt.subplots(figsize=(7, 4.5))
domain_agg = (
    df.groupby("domain")["log_view_per_day"]
      .agg(["mean", "sem"])
      .reindex(DOMAIN_ORDER)
      .reset_index()
)
bars = ax.bar(
    domain_agg["domain"],
    domain_agg["mean"],
    yerr=domain_agg["sem"],
    color=PALETTE_DOMAIN,
    width=0.55,
    capsize=4,
    error_kw={"elinewidth": 1.2, "ecolor": ZJU_GRAY},
    zorder=3,
)
ax.set_xlabel("知识领域", labelpad=6)
ax.set_ylabel("平均 log(日均播放量)", labelpad=6)
ax.set_title("图1  不同知识领域视频传播效果比较", pad=10, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7, zorder=0)
# 数值标签
for bar, (_, row) in zip(bars, domain_agg.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row["sem"] + 0.02,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8, color=ZJU_GRAY)
ax.set_ylim(bottom=0)
# 样本量标注
for bar, dom in zip(bars, DOMAIN_ORDER):
    n = (df["domain"] == dom).sum()
    ax.text(bar.get_x() + bar.get_width() / 2,
            0.02, f"N={n}",
            ha="center", va="bottom", fontsize=7.5, color="white",
            fontweight="bold")
fig.text(0.12, -0.03,
         "注：误差棒为标准误（SE）；N 为各领域视频样本量。",
         fontsize=8, color=ZJU_GRAY, style="italic")
plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, "domain_spread_bar.png")
fig.savefig(fig1_path)
plt.close(fig)
log.info(f"  图1 已保存 → {fig1_path}")

# ─ 图2：complementarity_type 与平均传播效果柱状图 ─
TYPE_ORDER = [
    "high_synergy", "complementary_explanation", "curiosity_contrast",
    "redundant_reinforcement", "emotional_conflict", "low_relatedness"
]
TYPE_LABELS = {
    "high_synergy":              "高协同",
    "complementary_explanation": "互补阐释",
    "curiosity_contrast":        "好奇对比",
    "redundant_reinforcement":   "冗余强化",
    "emotional_conflict":        "情感冲突",
    "low_relatedness":           "低关联",
}
type_agg = (
    df.groupby("complementarity_type")["communication_effect"]
      .agg(["mean", "sem", "count"])
      .reindex(TYPE_ORDER)
      .reset_index()
)
colors_type = [ZJU_BLUE, ZJU_GREEN, ZJU_GOLD, ZJU_LGRAY, ZJU_RED, ZJU_GRAY]

fig2, ax2 = plt.subplots(figsize=(8.5, 4.5))
bars2 = ax2.bar(
    [TYPE_LABELS.get(t, t) for t in type_agg["complementarity_type"]],
    type_agg["mean"],
    yerr=type_agg["sem"],
    color=colors_type[:len(type_agg)],
    width=0.6,
    capsize=4,
    error_kw={"elinewidth": 1.2, "ecolor": "#444444"},
    zorder=3,
)
ax2.set_xlabel("双模态互补类型", labelpad=6)
ax2.set_ylabel("平均传播效果综合指数", labelpad=6)
ax2.set_title("图2  不同双模态协同类型的视频传播效果比较", pad=10, fontweight="bold")
ax2.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7, zorder=0)
for bar, (_, row) in zip(bars2, type_agg.iterrows()):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + row["sem"] + 0.02,
             f"{bar.get_height():.3f}",
             ha="center", va="bottom", fontsize=8, color=ZJU_GRAY)
for bar, (_, row) in zip(bars2, type_agg.iterrows()):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             0.02, f"N={int(row['count'])}",
             ha="center", va="bottom", fontsize=7.5, color="white", fontweight="bold")
# 单尾 ANOVA p 值标注
grp_vals = [df[df["complementarity_type"] == t]["communication_effect"].dropna().values
            for t in TYPE_ORDER if t in df["complementarity_type"].values]
if len(grp_vals) >= 2:
    f_stat, f_p = stats.f_oneway(*grp_vals)
    ax2.text(0.98, 0.97, f"单因素 ANOVA: F={f_stat:.2f}, p{'<0.001' if f_p<0.001 else f'={f_p:.3f}'}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8, color=ZJU_GRAY,
             style="italic")
fig2.text(0.12, -0.03,
          "注：误差棒为标准误（SE）；N 为各类型视频样本量；括号内为英文变量名。",
          fontsize=8, color=ZJU_GRAY, style="italic")
plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, "multimodal_type_spread_bar.png")
fig2.savefig(fig2_path)
plt.close(fig2)
log.info(f"  图2 已保存 → {fig2_path}")

# ─ 图3：高低传播组 image_title_fit 箱线图（median & q 双面板）─
fig3, axes3 = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

for ax_i, (group_col, title_suffix, df_sub) in enumerate([
    ("median_group", "（中位数分组）", df),
    ("q_group",      "（四分位分组）", df[df["q_group"] != "中间"]),
]):
    ax = axes3[ax_i]
    groups = ["高传播组", "低传播组"]
    data_plot = [df_sub[df_sub[group_col] == g]["image_title_fit"].dropna().values
                 for g in groups]
    bplot = ax.boxplot(
        data_plot,
        labels=groups,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=2.5,
                        markerfacecolor=ZJU_GRAY, alpha=0.3),
        widths=0.45,
    )
    for patch, color in zip(bplot["boxes"], [ZJU_BLUE, ZJU_GOLD]):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    # 叠加均值点
    for xi, d in enumerate(data_plot, 1):
        ax.scatter(xi, np.mean(d), marker="D", color="white",
                   s=30, zorder=5, edgecolors="black", linewidths=0.8)
    # Mann-Whitney U 检验 p 值
    mw_s, mw_p = stats.mannwhitneyu(data_plot[0], data_plot[1], alternative="two-sided")
    sig = sig_stars(mw_p)
    y_max = max(np.percentile(d, 99) for d in data_plot if len(d))
    ax.plot([1, 1, 2, 2], [y_max, y_max + 0.02, y_max + 0.02, y_max],
            linewidth=1.2, color=ZJU_GRAY)
    ax.text(1.5, y_max + 0.025,
            f"MWU: p={'<0.001' if mw_p < 0.001 else f'{mw_p:.3f}'} {sig}",
            ha="center", va="bottom", fontsize=8.5, color=ZJU_GRAY)
    ax.set_title(f"语义一致性（image_title_fit）{title_suffix}", fontsize=9.5)
    ax.set_ylabel("语义一致性得分" if ax_i == 0 else "", labelpad=6)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("传播组别", labelpad=6)

fig3.suptitle("图3  高低传播组封面—标题语义一致性比较", fontsize=12, fontweight="bold", y=1.01)
fig3.text(0.12, -0.04,
          "注：◆ 表示均值；箱体上下边界为四分位数 Q1/Q3；虚线为中位数水平。",
          fontsize=8, color=ZJU_GRAY, style="italic")
plt.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, "high_low_image_title_fit_boxplot.png")
fig3.savefig(fig3_path)
plt.close(fig3)
log.info(f"  图3 已保存 → {fig3_path}")

# ─ 图4：高低传播组 title_cover_text_overlap 箱线图 ─
fig4, axes4 = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

for ax_i, (group_col, title_suffix, df_sub) in enumerate([
    ("median_group", "（中位数分组）", df),
    ("q_group",      "（四分位分组）", df[df["q_group"] != "中间"]),
]):
    ax = axes4[ax_i]
    groups = ["高传播组", "低传播组"]
    data_plot = [df_sub[df_sub[group_col] == g]["title_cover_text_overlap"].dropna().values
                 for g in groups]
    bplot = ax.boxplot(
        data_plot,
        labels=groups,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=2.5,
                        markerfacecolor=ZJU_GRAY, alpha=0.3),
        widths=0.45,
    )
    for patch, color in zip(bplot["boxes"], [ZJU_BLUE, ZJU_GOLD]):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    for xi, d in enumerate(data_plot, 1):
        ax.scatter(xi, np.mean(d), marker="D", color="white",
                   s=30, zorder=5, edgecolors="black", linewidths=0.8)
    mw_s, mw_p = stats.mannwhitneyu(data_plot[0], data_plot[1], alternative="two-sided")
    sig = sig_stars(mw_p)
    y_max = max(np.percentile(d, 99) for d in data_plot if len(d))
    ax.plot([1, 1, 2, 2], [y_max, y_max + 0.02, y_max + 0.02, y_max],
            linewidth=1.2, color=ZJU_GRAY)
    ax.text(1.5, y_max + 0.025,
            f"MWU: p={'<0.001' if mw_p < 0.001 else f'{mw_p:.3f}'} {sig}",
            ha="center", va="bottom", fontsize=8.5, color=ZJU_GRAY)
    ax.set_title(f"文字重叠度（title_cover_text_overlap）{title_suffix}", fontsize=9.5)
    ax.set_ylabel("文字重叠度得分" if ax_i == 0 else "", labelpad=6)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("传播组别", labelpad=6)

fig4.suptitle("图4  高低传播组标题—封面文字重叠度比较", fontsize=12, fontweight="bold", y=1.01)
fig4.text(0.12, -0.04,
          "注：◆ 表示均值；箱体上下边界为四分位数 Q1/Q3。",
          fontsize=8, color=ZJU_GRAY, style="italic")
plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, "high_low_title_overlap_boxplot.png")
fig4.savefig(fig4_path)
plt.close(fig4)
log.info(f"  图4 已保存 → {fig4_path}")

# ─ 图5：高低传播组标题特征对比图（均值差异水平条形图）─
title_cont_vars = [
    ("title_char_len",            "标题字符数"),
    ("title_word_count",          "标题词数"),
    ("title_sentiment_intensity", "情感强度"),
    ("curiosity_marker_count",    "好奇心标记词数"),
    ("practical_value_word_count","实用价值词数"),
]
title_bin_vars = [
    ("title_has_number",          "含数字(占比)"),
    ("title_has_practical_value", "含实用价值词(占比)"),
]

# 连续变量：Z 标准化后比较均值差异
fig5, axes5 = plt.subplots(1, 2, figsize=(11, 5.5))

# 左：连续变量均值（标准化）
ax5l = axes5[0]
rows_bar = []
for var, label in title_cont_vars:
    if var not in df.columns: continue
    hi = df[df["median_group"] == "高传播组"][var].dropna()
    lo = df[df["median_group"] == "低传播组"][var].dropna()
    hi_mean = hi.mean()
    lo_mean = lo.mean()
    _, t_p = stats.ttest_ind(hi, lo, equal_var=False)
    _, mw_p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
    rows_bar.append({
        "label": label, "高传播组": hi_mean, "低传播组": lo_mean,
        "stars": sig_stars(min(t_p, mw_p))
    })

bar_df = pd.DataFrame(rows_bar)
x = np.arange(len(bar_df))
width = 0.35
b1 = ax5l.barh(x + width/2, bar_df["高传播组"], height=width,
               color=ZJU_BLUE, label="高传播组", alpha=0.85)
b2 = ax5l.barh(x - width/2, bar_df["低传播组"], height=width,
               color=ZJU_GOLD, label="低传播组", alpha=0.85)
ax5l.set_yticks(x)
ax5l.set_yticklabels(bar_df["label"], fontsize=9)
ax5l.set_xlabel("均值", labelpad=6)
ax5l.set_title("(a) 连续型标题特征", fontsize=10)
ax5l.legend(loc="lower right", fontsize=8.5)
ax5l.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)
# 显著性标注
x_max = max(bar_df["高传播组"].max(), bar_df["低传播组"].max())
for yi, row in bar_df.iterrows():
    if row["stars"]:
        ax5l.text(x_max * 1.02, yi, row["stars"],
                  va="center", ha="left", fontsize=10, color=ZJU_RED)
ax5l.set_xlim(right=x_max * 1.15)

# 右：二值变量占比对比
ax5r = axes5[1]
rows_bar2 = []
for var, label in title_bin_vars:
    if var not in df.columns: continue
    hi = df[df["median_group"] == "高传播组"][var].dropna()
    lo = df[df["median_group"] == "低传播组"][var].dropna()
    ct = pd.crosstab(df["median_group"], df[var].fillna(0).astype(int))
    if ct.shape == (2, 2):
        chi2, chi_p, _, _ = stats.chi2_contingency(ct)
    else:
        chi_p = 1.0
    rows_bar2.append({
        "label": label,
        "高传播组": hi.mean() * 100,
        "低传播组": lo.mean() * 100,
        "stars": sig_stars(chi_p)
    })
bar_df2 = pd.DataFrame(rows_bar2)
if len(bar_df2) > 0:
    x2 = np.arange(len(bar_df2))
    ax5r.barh(x2 + width/2, bar_df2["高传播组"], height=width,
              color=ZJU_BLUE, label="高传播组", alpha=0.85)
    ax5r.barh(x2 - width/2, bar_df2["低传播组"], height=width,
              color=ZJU_GOLD, label="低传播组", alpha=0.85)
    ax5r.set_yticks(x2)
    ax5r.set_yticklabels(bar_df2["label"], fontsize=9)
    ax5r.set_xlabel("占比 (%)", labelpad=6)
    ax5r.set_title("(b) 二值型标题特征", fontsize=10)
    ax5r.legend(loc="lower right", fontsize=8.5)
    ax5r.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)
    x_max2 = max(bar_df2["高传播组"].max(), bar_df2["低传播组"].max())
    for yi, row in bar_df2.iterrows():
        if row["stars"]:
            ax5r.text(x_max2 * 1.02, yi, row["stars"],
                      va="center", ha="left", fontsize=10, color=ZJU_RED)
    ax5r.set_xlim(right=x_max2 * 1.2)
    ax5r.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

fig5.suptitle("图5  高低传播组标题文本特征比较", fontsize=12, fontweight="bold")
fig5.text(0.12, -0.03,
          "注：分组依据为 log_view_per_day 中位数；* p<0.05，** p<0.01，*** p<0.001。",
          fontsize=8, color=ZJU_GRAY, style="italic")
plt.tight_layout()
fig5_path = os.path.join(FIGURES_DIR, "high_low_title_features_bar.png")
fig5.savefig(fig5_path)
plt.close(fig5)
log.info(f"  图5 已保存 → {fig5_path}")

# ── 20. 质量检查报告 ──────────────────────────────────────────────────────
log.info("\n[6-12] 生成质量检查报告 ...")

with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write("\n" + "=" * 70 + "\n")
    f.write("质量检查报告\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"1. 建模数据集基本信息\n")
    f.write(f"   总行数：{len(df)}\n")
    f.write(f"   总列数：{df.shape[1]}\n")
    f.write(f"   bvid 唯一值：{df['bvid'].nunique()}\n")

    f.write(f"\n2. 缺失值情况\n")
    null_sum = df.isnull().sum()
    null_cols2 = null_sum[null_sum > 0].sort_values(ascending=False)
    if len(null_cols2):
        for col, cnt in null_cols2.items():
            f.write(f"   {col}: {cnt} ({cnt/len(df)*100:.2f}%)\n")
    else:
        f.write("   无缺失值\n")

    f.write(f"\n3. 分组检验样本量\n")
    f.write(f"   median_group 高传播组: {(df['median_group']=='高传播组').sum()}\n")
    f.write(f"   median_group 低传播组: {(df['median_group']=='低传播组').sum()}\n")
    f.write(f"   q_group 高传播组(top 25%): {(df['q_group']=='高传播组').sum()}\n")
    f.write(f"   q_group 低传播组(bot 25%): {(df['q_group']=='低传播组').sum()}\n")
    f.write(f"   q_group 中间(排除): {(df['q_group']=='中间').sum()}\n")

    f.write(f"\n4. 异常值检测（基于 log_view_per_day）\n")
    z = np.abs(stats.zscore(df["log_view_per_day"].dropna()))
    n_outlier = (z > 3).sum()
    f.write(f"   |Z|>3 异常值数：{n_outlier} ({n_outlier/len(df)*100:.2f}%)\n")

    f.write(f"\n5. 待补全变量\n")
    missing_expected = [c for c in EXPECTED_COLS if df[c].isna().all()]
    if missing_expected:
        for c in missing_expected:
            f.write(f"   {c}：全部缺失，需在图像分类步骤（第四步）补全\n")
    else:
        f.write("   无\n")

    f.write(f"\n6. 输出文件清单\n")
    for fp in [out_model, desc_path, diff_path,
               fig1_path, fig2_path, fig3_path, fig4_path, fig5_path]:
        exists = "✓" if os.path.exists(fp) else "✗"
        f.write(f"   [{exists}] {fp}\n")

    f.write(f"\n执行完成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

log.info(f"  日志/质量报告 → {LOG_FILE}")

# ── 21. 完成汇总 ──────────────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("第六步执行完毕，输出文件汇总：")
output_files = {
    "建模数据集":      out_model,
    "描述性统计表":    desc_path,
    "差异检验表":      diff_path,
    "图1-领域传播":    fig1_path,
    "图2-双模态类型":  fig2_path,
    "图3-语义一致性箱线图": fig3_path,
    "图4-文字重叠度箱线图": fig4_path,
    "图5-标题特征对比图":   fig5_path,
    "质量报告日志":    LOG_FILE,
}
for label, path in output_files.items():
    status = "✓" if os.path.exists(path) else "✗ 未生成"
    log.info(f"  [{status}] {label}: {path}")
log.info("=" * 70)
