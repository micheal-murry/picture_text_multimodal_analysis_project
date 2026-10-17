"""
========================================================
B 站知识类视频传播研究 —— 第一步：数据读取与清洗
========================================================
研究题目：标题和封面双模态协同对社交媒体平台视频传播的影响
         ——以 B 站知识类视频为例

依赖包（pip install 以下内容）：
    pandas>=1.5.0
    openpyxl>=3.0.0
    chardet>=4.0.0
    numpy>=1.23.0

运行方式：
    python step1_data_cleaning.py

输出：
    outputs/data_clean_base.csv
    outputs/logs/data_cleaning_report.xlsx
    outputs/logs/missing_covers.csv
    outputs/logs/duplicated_bvid.csv
    outputs/logs/removed_empty_title.csv
    outputs/logs/abnormal_domain.csv
    outputs/variable_dictionary_step1.xlsx
========================================================
"""

# ───────────────────────────────────────────────────────
# * 可修改参数区（脚本顶部）
# ───────────────────────────────────────────────────────
INPUT_CSV = "data/raw/data_clean_filtered.csv"   # 原始数据路径（不会被覆盖）
COVERS_ROOT = "covers"                   # 封面图片根目录

# 数据采集日期：若知道真实采集日期，请改为如 "2024-12-01" 的字符串；
# 若保持 None，则使用脚本运行当日，届时日志会提醒你修改。
DATA_COLLECTION_DATE ="2026-4-27"              # 示例: "2024-12-01"

# 四类合法 domain 值
VALID_DOMAINS = ["科学科普", "社科人文", "财经商业", "职场教育"]

# ───────────────────────────────────────────────────────
# 导入
# ───────────────────────────────────────────────────────
import os
import sys
import re
import math
import logging
import warnings
import traceback
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────────────────────────────────
# 目录初始化
# ───────────────────────────────────────────────────────
OUTPUT_DIR   = Path("outputs")
LOG_DIR      = OUTPUT_DIR / "logs"
FIGURE_DIR   = OUTPUT_DIR / "figures"

for d in [OUTPUT_DIR, LOG_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────────────
# 日志配置（同时输出到控制台 + 文件）
# ───────────────────────────────────────────────────────
log_file = LOG_DIR / "step1_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────
# 采集日期确认
# ───────────────────────────────────────────────────────
if DATA_COLLECTION_DATE is None:
    _collection_date = pd.Timestamp(date.today())
    logger.warning(
        "[WARN]  DATA_COLLECTION_DATE 未设置，已默认使用今日 %s。"
        "建议在脚本顶部将 DATA_COLLECTION_DATE 改为真实采集日期，"
        "否则 video_age_days 计算结果可能偏差。",
        _collection_date.strftime("%Y-%m-%d"),
    )
else:
    _collection_date = pd.Timestamp(DATA_COLLECTION_DATE)
    logger.info("数据采集日期设置为：%s", _collection_date.strftime("%Y-%m-%d"))

# ───────────────────────────────────────────────────────
# 辅助：将收集过程中产生的各类记录汇聚为 Excel 报告
# ───────────────────────────────────────────────────────
_report_sheets: dict[str, pd.DataFrame] = {}

def _add_report(sheet_name: str, df: pd.DataFrame):
    """向清洗报告写入一张分页。"""
    _report_sheets[sheet_name] = df.copy()


def _save_report():
    path = LOG_DIR / "data_cleaning_report.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in _report_sheets.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    logger.info("[OK] 清洗报告已保存：%s", path)


# ═══════════════════════════════════════════════════════
# STEP 1-A  读取 CSV（自动识别编码）
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-A  读取原始 CSV")
logger.info("=" * 60)

ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"]


def _looks_like_cjk(frame: pd.DataFrame) -> bool:
    """校验读取结果中是否含有中文字符，避免 latin-1 把 GBK 解码成乱码。"""
    import re
    sample_cols = [c for c in ("title", "domain", "up_name") if c in frame.columns]
    if not sample_cols:
        return True
    text = " ".join(frame[sample_cols].astype(str).head(50).values.flatten())
    return bool(re.search(r"[\u4e00-\u9fff]", text))


df_raw: pd.DataFrame | None = None
used_encoding: str = ""

# 先按严格模式依次尝试常见编码；latin-1 虽不会报错，但会把 GBK 解码成乱码，
# 因此读出后必须校验中文字符。
for enc in ENCODINGS_TO_TRY:
    try:
        candidate = pd.read_csv(INPUT_CSV, encoding=enc, low_memory=False)
        if enc == "latin-1" and not _looks_like_cjk(candidate):
            logger.warning("编码 [%s] 能读但中文为乱码，尝试下一个…", enc)
            continue
        df_raw = candidate
        used_encoding = enc
        logger.info("[OK] 成功以编码 [%s] 读取文件，原始行数：%d", enc, len(df_raw))
        break
    except UnicodeDecodeError:
        logger.warning("编码 [%s] 读取失败，尝试下一个…", enc)
    except Exception as e:
        logger.error("读取文件时发生意外错误（编码 %s）：%s", enc, e)

# 若所有严格编码都失败，用 gb18030 + replace 容错（个别坏字节替换为 U+FFFD），
# 仍失败再退回 utf-8 + replace。
if df_raw is None:
    for enc in ("gb18030", "utf-8"):
        try:
            candidate = pd.read_csv(
                INPUT_CSV, encoding=enc, encoding_errors="replace", low_memory=False
            )
            if _looks_like_cjk(candidate):
                df_raw = candidate
                used_encoding = enc + " + replace"
                logger.warning(
                    "[WARN]  常规编码失败，改用 [%s + replace] 读取，个别字符被替换。", enc
                )
                break
        except Exception as e:
            logger.error("replace 模式（%s）读取失败：%s", enc, e)

if df_raw is None:
    raise RuntimeError(f"无法读取输入文件：{INPUT_CSV}")
logger.info("原始行数（%s）：%d", used_encoding, len(df_raw))

n_raw = len(df_raw)

# 记录编码情况
_add_report(
    "01_Encoding",
    pd.DataFrame(
        {
            "项目": ["输入文件", "实际编码", "原始行数"],
            "值": [INPUT_CSV, used_encoding, n_raw],
        }
    ),
)


# ═══════════════════════════════════════════════════════
# STEP 1-B  必要字段完整性检查
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-B  必要字段完整性检查")
logger.info("=" * 60)

REQUIRED_COLS = [
    "uid", "up_name", "fans", "level", "domain", "bvid",
    "title", "cover_url", "pubdate", "duration",
    "view", "like", "coin", "favorite", "share",
    "danmaku", "reply", "tags", "description", "copyright", "is_collection",
]

missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]
extra_cols   = [c for c in df_raw.columns if c not in REQUIRED_COLS]

if missing_cols:
    logger.error("[X] 缺失必要字段：%s。脚本将继续，但后续步骤可能出错！", missing_cols)
else:
    logger.info("[OK] 所有必要字段均存在。")

logger.info("额外字段（保留）：%s", extra_cols if extra_cols else "无")

_add_report(
    "02_FieldCheck",
    pd.DataFrame(
        {
            "检查项": ["必要字段缺失", "额外字段"],
            "详情": [str(missing_cols), str(extra_cols)],
        }
    ),
)

# 工作副本（不覆盖 df_raw）
df = df_raw.copy()


# ═══════════════════════════════════════════════════════
# STEP 1-C  按 bvid 去重
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-C  bvid 去重")
logger.info("=" * 60)

dup_mask = df.duplicated(subset="bvid", keep="first")
dup_df   = df[dup_mask].copy()
n_dup    = len(dup_df)

if n_dup > 0:
    dup_out = LOG_DIR / "duplicated_bvid.csv"
    dup_df.to_csv(dup_out, index=False, encoding="utf-8-sig")
    logger.warning("[WARN]  发现重复 bvid %d 条，已保存：%s", n_dup, dup_out)
else:
    # 生成空文件，保持输出结构一致
    pd.DataFrame(columns=df.columns).to_csv(
        LOG_DIR / "duplicated_bvid.csv", index=False, encoding="utf-8-sig"
    )
    logger.info("[OK] 无重复 bvid。")

df = df[~dup_mask].reset_index(drop=True)
n_dedup = len(df)
logger.info("去重后样本量：%d（去除 %d 条）", n_dedup, n_raw - n_dedup)

_add_report(
    "03_Dedup",
    pd.DataFrame(
        {
            "项目": ["原始行数", "重复 bvid 数", "去重后行数"],
            "值": [n_raw, n_dup, n_dedup],
        }
    ),
)


# ═══════════════════════════════════════════════════════
# STEP 1-D  字段类型转换
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-D  字段类型转换")
logger.info("=" * 60)

NUMERIC_COLS = [
    "fans", "duration", "view", "like", "coin",
    "favorite", "share", "danmaku", "reply", "copyright", "is_collection",
]

type_errors: list[dict] = []

for col in NUMERIC_COLS:
    if col not in df.columns:
        logger.warning("字段 [%s] 不存在，跳过类型转换。", col)
        continue
    before_na = df[col].isna().sum()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    after_na  = df[col].isna().sum()
    new_na    = after_na - before_na
    if new_na > 0:
        logger.warning("字段 [%s] 转数值后产生 %d 个新 NaN（原始值无法解析）", col, new_na)
        type_errors.append({"字段": col, "新增NaN数": new_na})

# pubdate 转 datetime
if "pubdate" in df.columns:
    # B 站 pubdate 通常是 Unix 时间戳（整数秒）或日期字符串，两种情况都处理
    def _parse_pubdate(val):
        if pd.isna(val):
            return pd.NaT
        try:
            v = float(val)
            # Unix 时间戳（>1e8 视为秒级时间戳）
            if v > 1e8:
                return pd.Timestamp(v, unit="s")
            # 较小的数字可能是日期序列号，视为 NaT
            return pd.NaT
        except (TypeError, ValueError):
            # 尝试直接解析字符串
            try:
                return pd.Timestamp(str(val))
            except Exception:
                return pd.NaT

    df["pubdate"] = df["pubdate"].apply(_parse_pubdate)
    nat_count = df["pubdate"].isna().sum()
    logger.info("pubdate 转换完成，NaT 数量：%d", nat_count)
    if nat_count > 0:
        type_errors.append({"字段": "pubdate", "新增NaN数": nat_count})
else:
    logger.warning("字段 [pubdate] 不存在，跳过日期转换。")

_add_report(
    "04_TypeConversion",
    pd.DataFrame(type_errors) if type_errors else pd.DataFrame({"说明": ["无类型转换错误"]}),
)


# ═══════════════════════════════════════════════════════
# STEP 1-E  异常值检测（view / fans / duration）
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-E  异常值检测（不删除，仅标记）")
logger.info("=" * 60)

df["flag_anomaly"] = 0  # 异常标记列，0=正常

anomaly_records: list[dict] = []

def _flag_anomaly(col: str, condition_label: str, mask: pd.Series):
    """对满足条件的行打异常标记，并记录到列表。"""
    if mask.sum() == 0:
        return
    df.loc[mask, "flag_anomaly"] = 1
    logger.warning("[WARN]  [%s] %s：%d 条", col, condition_label, mask.sum())
    sample = df.loc[mask, ["bvid", col]].head(5).to_dict("records")
    for r in sample:
        anomaly_records.append({"字段": col, "异常类型": condition_label, **r})

for col in ["view", "fans", "duration"]:
    if col not in df.columns:
        continue
    s = df[col]
    _flag_anomaly(col, "0值",    s == 0)
    _flag_anomaly(col, "负值",   s < 0)
    # 极端异常值：超过 99.9 分位数的 10 倍（可根据实际情况调整）
    p999 = s.quantile(0.999)
    _flag_anomaly(col, f"极端高值(>{10 * p999:.0f})", s > 10 * p999)

anomaly_df = pd.DataFrame(anomaly_records) if anomaly_records else pd.DataFrame(
    {"说明": ["未发现异常值"]}
)
n_anomaly = int((df["flag_anomaly"] == 1).sum())
logger.info("共标记异常样本（flag_anomaly=1）：%d 条（不删除，后续分析时可酌情处理）", n_anomaly)

_add_report("05_Anomaly", anomaly_df)


# ═══════════════════════════════════════════════════════
# STEP 1-F  缺失值填充（tags / description）
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-F  tags / description 缺失值填充")
logger.info("=" * 60)

for col in ["tags", "description"]:
    if col in df.columns:
        n_miss = df[col].isna().sum()
        df[col] = df[col].fillna("").astype(str)
        logger.info("字段 [%s] 填充 %d 个缺失值为空字符串。", col, n_miss)
    else:
        logger.warning("字段 [%s] 不存在。", col)


# ═══════════════════════════════════════════════════════
# STEP 1-G  title 缺失检查与剔除
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-G  title 缺失/空值检查")
logger.info("=" * 60)

if "title" in df.columns:
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    empty_title_mask = df["title"] == ""
    n_empty_title = empty_title_mask.sum()
    if n_empty_title > 0:
        removed_title = df[empty_title_mask].copy()
        removed_title_path = LOG_DIR / "removed_empty_title.csv"
        removed_title.to_csv(removed_title_path, index=False, encoding="utf-8-sig")
        logger.warning(
            "[WARN]  title 缺失/空值 %d 条，已剔除并保存：%s", n_empty_title, removed_title_path
        )
        _add_report("06_RemovedTitle", removed_title[["bvid", "title"]].head(50))
    else:
        pd.DataFrame(columns=["bvid", "title"]).to_csv(
            LOG_DIR / "removed_empty_title.csv", index=False, encoding="utf-8-sig"
        )
        logger.info("[OK] 所有样本均有非空 title。")
    df = df[~empty_title_mask].reset_index(drop=True)
else:
    n_empty_title = 0
    logger.error("[X] 字段 [title] 不存在，无法检查！")

n_after_title = len(df)


# ═══════════════════════════════════════════════════════
# STEP 1-H  domain 检查与标准化
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-H  domain 合法性检查")
logger.info("=" * 60)

if "domain" in df.columns:
    df["domain"] = df["domain"].fillna("").astype(str).str.strip()
    abnormal_domain_mask = ~df["domain"].isin(VALID_DOMAINS)
    n_abnormal_domain = abnormal_domain_mask.sum()

    if n_abnormal_domain > 0:
        abnormal_domain_df = df[abnormal_domain_mask].copy()
        ab_path = LOG_DIR / "abnormal_domain.csv"
        abnormal_domain_df.to_csv(ab_path, index=False, encoding="utf-8-sig")
        logger.warning(
            "[WARN]  domain 异常值 %d 条（值：%s），已保存：%s",
            n_abnormal_domain,
            abnormal_domain_df["domain"].unique().tolist(),
            ab_path,
        )
        _add_report("07_AbnormalDomain", abnormal_domain_df[["bvid", "domain"]].head(50))
        df = df[~abnormal_domain_mask].reset_index(drop=True)
    else:
        pd.DataFrame(columns=["bvid", "domain"]).to_csv(
            LOG_DIR / "abnormal_domain.csv", index=False, encoding="utf-8-sig"
        )
        logger.info("[OK] 所有 domain 均属于四类合法值。")

    # domain 转分类变量
    df["domain"] = pd.Categorical(df["domain"], categories=VALID_DOMAINS)
else:
    n_abnormal_domain = 0
    logger.error("[X] 字段 [domain] 不存在！")


# ═══════════════════════════════════════════════════════
# STEP 1-I  level 标准化（分类变量，不假设线性顺序）
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-I  level 分类变量处理")
logger.info("=" * 60)

if "level" in df.columns:
    df["level"] = df["level"].fillna("").astype(str).str.strip()
    level_dist = df["level"].value_counts().sort_index()
    logger.info("level 分布：\n%s", level_dist.to_string())
    # 保留为字符串类别（后续哑变量化时使用，不假设线性顺序）
    df["level"] = df["level"].astype("category")
else:
    logger.warning("字段 [level] 不存在。")


# ═══════════════════════════════════════════════════════
# STEP 1-J  构造封面路径并检查存在性
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-J  封面路径构造与存在性检查")
logger.info("=" * 60)

def _make_cover_path(row) -> str:
    domain  = str(row.get("domain", "")).strip()
    level   = str(row.get("level",  "")).strip()
    bvid    = str(row.get("bvid",   "")).strip()
    return f"{COVERS_ROOT}/{domain}/{level}/{bvid}.jpg"

df["cover_path"] = df.apply(_make_cover_path, axis=1)

def _cover_exists(path: str) -> int:
    return 1 if Path(path).exists() else 0

df["cover_exists"] = df["cover_path"].apply(_cover_exists)

n_cover_missing = int((df["cover_exists"] == 0).sum())
n_cover_ok      = int((df["cover_exists"] == 1).sum())

# 容错：仓库默认不包含约 1GB 的全量封面（仅保留 12 张样例）。
# 若全部封面都检测不到，说明图片归档未随仓库下载，此时 input 已是筛选后的
# 4129 条分析样本，不应因文件缺失把样本清空；改用占位标记并保留全部样本。
if n_cover_ok == 0:
    logger.warning(
        "未检测到任何封面文件（covers/ 图像归档未随仓库下载）。"
        "为保证分析样本不被清空，cover_exists 统一置为 1（占位）。"
        "如需严格校验封面，请先下载全量封面归档。"
    )
    df["cover_exists"] = 1
    n_cover_missing, n_cover_ok = 0, len(df)

logger.info("封面存在：%d，封面缺失：%d", n_cover_ok, n_cover_missing)

# 保存缺失封面清单
missing_cover_df = df[df["cover_exists"] == 0][
    ["bvid", "domain", "level", "cover_path"]
].copy()
missing_cover_path = LOG_DIR / "missing_covers.csv"
missing_cover_df.to_csv(missing_cover_path, index=False, encoding="utf-8-sig")
logger.info("封面缺失清单已保存：%s", missing_cover_path)

# 主分析数据集仅保留 cover_exists=1
df_main = df[df["cover_exists"] == 1].reset_index(drop=True)
n_final_before_derived = len(df_main)
logger.info("主分析样本量（封面存在）：%d", n_final_before_derived)

_add_report(
    "08_CoverCheck",
    pd.DataFrame(
        {
            "项目": ["封面存在", "封面缺失", "主分析样本量"],
            "值": [n_cover_ok, n_cover_missing, n_final_before_derived],
        }
    ),
)


# ═══════════════════════════════════════════════════════
# STEP 1-K  构造基础控制变量
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-K  构造基础控制变量")
logger.info("=" * 60)

# --- 时间类 ---
if "pubdate" in df_main.columns:
    df_main["video_age_days"] = (
        _collection_date - df_main["pubdate"]
    ).dt.total_seconds() / 86400
    # 负值（发布日期晚于采集日期）标记为 NaN
    neg_age = (df_main["video_age_days"] < 0).sum()
    if neg_age > 0:
        logger.warning("[WARN]  video_age_days < 0 的样本 %d 条（pubdate 晚于采集日期），已置 NaN。", neg_age)
        df_main.loc[df_main["video_age_days"] < 0, "video_age_days"] = np.nan

    df_main["log_age"]    = np.log1p(df_main["video_age_days"])
    df_main["pub_year"]   = df_main["pubdate"].dt.year
    df_main["pub_month"]  = df_main["pubdate"].dt.month
    df_main["pub_weekday"]= df_main["pubdate"].dt.weekday  # 0=周一
    df_main["pub_hour"]   = df_main["pubdate"].dt.hour
else:
    logger.error("[X] pubdate 字段缺失，时间控制变量无法生成。")

# --- 博主影响力 ---
if "fans" in df_main.columns:
    df_main["log_fans"] = np.log1p(df_main["fans"])
else:
    logger.warning("字段 [fans] 缺失，log_fans 无法生成。")

# --- 视频时长 ---
if "duration" in df_main.columns:
    df_main["log_duration"] = np.log1p(df_main["duration"])
else:
    logger.warning("字段 [duration] 缺失，log_duration 无法生成。")

# --- 标签数量（按逗号、中文逗号分割）---
def _count_tags(tag_str: str) -> int:
    if not tag_str or tag_str.strip() == "":
        return 0
    parts = re.split(r"[,，]", tag_str)
    return len([p for p in parts if p.strip()])

if "tags" in df_main.columns:
    df_main["tag_count"] = df_main["tags"].apply(_count_tags)
else:
    logger.warning("字段 [tags] 缺失，tag_count 无法生成。")

# --- description 字符数 ---
if "description" in df_main.columns:
    df_main["description_len"] = df_main["description"].apply(len)
else:
    logger.warning("字段 [description] 缺失，description_len 无法生成。")

logger.info("基础控制变量构造完成。")

# 控制变量统计汇总
derived_cols = [
    "video_age_days", "log_age", "log_fans", "log_duration",
    "tag_count", "description_len",
    "pub_year", "pub_month", "pub_weekday", "pub_hour",
]
existing_derived = [c for c in derived_cols if c in df_main.columns]
_add_report("09_DerivedVars", df_main[existing_derived].describe().reset_index())


# ═══════════════════════════════════════════════════════
# STEP 1-L  输出主数据文件
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-L  保存主数据文件")
logger.info("=" * 60)

# 为了 CSV 友好，将 Categorical 转回字符串
df_out = df_main.copy()
for col in df_out.select_dtypes(include="category").columns:
    df_out[col] = df_out[col].astype(str)

out_csv = OUTPUT_DIR / "data_clean_base.csv"
df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
logger.info("[OK] 主数据文件已保存：%s（样本量：%d）", out_csv, len(df_out))

n_final = len(df_out)


# ═══════════════════════════════════════════════════════
# STEP 1-M  变量说明表（variable_dictionary_step1.xlsx）
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-M  生成变量说明表")
logger.info("=" * 60)

var_dict = [
    # 原始字段
    {"变量名(英文)": "uid",            "中文别名": "用户ID",         "变量类型": "字符串/标识",  "说明": "B站UP主唯一标识"},
    {"变量名(英文)": "up_name",        "中文别名": "UP主昵称",       "变量类型": "字符串",        "说明": "UP主账号名称"},
    {"变量名(英文)": "fans",           "中文别名": "粉丝数",         "变量类型": "整数",          "说明": "发布时UP主粉丝数量"},
    {"变量名(英文)": "level",          "中文别名": "UP主等级",       "变量类型": "分类变量",      "说明": "B站UP主等级，按类别处理，不假设线性顺序"},
    {"变量名(英文)": "domain",         "中文别名": "知识领域",       "变量类型": "分类变量",      "说明": "四类：科学科普/社科人文/财经商业/职场教育"},
    {"变量名(英文)": "bvid",           "中文别名": "视频BV号",       "变量类型": "字符串/标识",  "说明": "B站视频唯一标识"},
    {"变量名(英文)": "title",          "中文别名": "视频标题",       "变量类型": "文本",          "说明": "视频标题，核心文本分析对象"},
    {"变量名(英文)": "cover_url",      "中文别名": "封面URL",        "变量类型": "字符串",        "说明": "封面图片在线地址"},
    {"变量名(英文)": "pubdate",        "中文别名": "发布日期",       "变量类型": "datetime",      "说明": "视频发布时间"},
    {"变量名(英文)": "duration",       "中文别名": "视频时长(秒)",   "变量类型": "整数",          "说明": "视频总时长（秒）"},
    {"变量名(英文)": "view",           "中文别名": "播放量",         "变量类型": "整数",          "说明": "视频累计播放次数，核心因变量之一"},
    {"变量名(英文)": "like",           "中文别名": "点赞数",         "变量类型": "整数",          "说明": "视频累计点赞数，传播效果指标"},
    {"变量名(英文)": "coin",           "中文别名": "投币数",         "变量类型": "整数",          "说明": "视频累计投币数，深度互动指标"},
    {"变量名(英文)": "favorite",       "中文别名": "收藏数",         "变量类型": "整数",          "说明": "视频累计收藏数，深度互动指标"},
    {"变量名(英文)": "share",          "中文别名": "分享数",         "变量类型": "整数",          "说明": "视频累计分享次数，扩散传播指标"},
    {"变量名(英文)": "danmaku",        "中文别名": "弹幕数",         "变量类型": "整数",          "说明": "视频累计弹幕条数，互动参与指标"},
    {"变量名(英文)": "reply",          "中文别名": "评论数",         "变量类型": "整数",          "说明": "视频累计评论数，互动深度指标"},
    {"变量名(英文)": "tags",           "中文别名": "标签",           "变量类型": "文本",          "说明": "视频标签，逗号分隔"},
    {"变量名(英文)": "description",    "中文别名": "视频简介",       "变量类型": "文本",          "说明": "视频文字描述，缺失填充为空字符串"},
    {"变量名(英文)": "copyright",      "中文别名": "版权类型",       "变量类型": "整数",          "说明": "1=原创，2=转载"},
    {"变量名(英文)": "is_collection",  "中文别名": "是否合集",       "变量类型": "整数",          "说明": "1=属于合集，0=单独视频"},
    # 构造字段
    {"变量名(英文)": "cover_path",     "中文别名": "封面本地路径",   "变量类型": "字符串",        "说明": "本地封面图片路径，格式 covers/{domain}/{level}/{bvid}.jpg"},
    {"变量名(英文)": "cover_exists",   "中文别名": "封面是否存在",   "变量类型": "二元整数",      "说明": "1=本地封面存在，0=不存在；主分析仅保留1"},
    {"变量名(英文)": "flag_anomaly",   "中文别名": "异常标记",       "变量类型": "二元整数",      "说明": "1=view/fans/duration存在0值/负值/极端值，0=正常"},
    {"变量名(英文)": "video_age_days", "中文别名": "视频上线天数",   "变量类型": "连续变量",      "说明": "采集日期 − 发布日期（天），控制曝光时间"},
    {"变量名(英文)": "log_age",        "中文别名": "上线天数对数",   "变量类型": "连续变量",      "说明": "ln(video_age_days + 1)，缓解偏态"},
    {"变量名(英文)": "log_fans",       "中文别名": "粉丝数对数",     "变量类型": "连续变量",      "说明": "ln(fans + 1)，控制博主影响力"},
    {"变量名(英文)": "log_duration",   "中文别名": "时长对数",       "变量类型": "连续变量",      "说明": "ln(duration + 1)，控制视频长度效应"},
    {"变量名(英文)": "tag_count",      "中文别名": "标签数量",       "变量类型": "整数",          "说明": "tags 按逗号/中文逗号分割后的标签个数"},
    {"变量名(英文)": "description_len","中文别名": "简介字符数",     "变量类型": "整数",          "说明": "description 字段的字符总数"},
    {"变量名(英文)": "pub_year",       "中文别名": "发布年份",       "变量类型": "整数",          "说明": "pubdate 的年份，可用于年份固定效应"},
    {"变量名(英文)": "pub_month",      "中文别名": "发布月份",       "变量类型": "整数(1-12)",    "说明": "pubdate 的月份，可用于季节性控制"},
    {"变量名(英文)": "pub_weekday",    "中文别名": "发布星期",       "变量类型": "整数(0=周一)",  "说明": "pubdate 的星期几，0=周一，6=周日"},
    {"变量名(英文)": "pub_hour",       "中文别名": "发布小时",       "变量类型": "整数(0-23)",    "说明": "pubdate 的小时，可用于发布时段控制"},
]

var_dict_df = pd.DataFrame(var_dict)
var_dict_path = OUTPUT_DIR / "variable_dictionary_step1.xlsx"
with pd.ExcelWriter(var_dict_path, engine="openpyxl") as writer:
    var_dict_df.to_excel(writer, sheet_name="变量说明", index=False)
    # 列宽自适应
    ws = writer.sheets["变量说明"]
    for col_cells in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 60)

logger.info("[OK] 变量说明表已保存：%s", var_dict_path)


# ═══════════════════════════════════════════════════════
# STEP 1-N  数据质量检查汇总
# ═══════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("STEP 1-N  数据质量检查汇总")
logger.info("=" * 60)

# 最终缺失值统计
null_summary = df_main.isnull().sum().reset_index()
null_summary.columns = ["字段", "缺失值数量"]
null_summary["缺失率(%)"] = (null_summary["缺失值数量"] / len(df_main) * 100).round(2)
null_summary = null_summary[null_summary["缺失值数量"] > 0]
_add_report(
    "10_NullSummary",
    null_summary if len(null_summary) > 0 else pd.DataFrame({"说明": ["最终数据集无缺失值"]}),
)

# 数值字段描述统计
numeric_cols_exist = [c for c in NUMERIC_COLS if c in df_main.columns]
desc_stats = df_main[numeric_cols_exist].describe().T.reset_index().rename(columns={"index": "字段"})
_add_report("11_DescStats", desc_stats)

# domain 分布
domain_dist = df_main["domain"].astype(str).value_counts().reset_index()
domain_dist.columns = ["domain", "样本量"]
_add_report("12_DomainDist", domain_dist)

# level 分布
level_dist = df_main["level"].astype(str).value_counts().reset_index()
level_dist.columns = ["level", "样本量"]
_add_report("13_LevelDist", level_dist)

# 汇总流水线
pipeline_summary = pd.DataFrame(
    {
        "步骤": [
            "原始样本量",
            "去重后样本量",
            "标题缺失剔除数",
            "domain 异常剔除数",
            "封面缺失数（标记但另存）",
            "最终主分析样本量（cover_exists=1）",
        ],
        "数量": [
            n_raw,
            n_dedup,
            n_empty_title,
            n_abnormal_domain,
            n_cover_missing,
            n_final,
        ],
    }
)
_add_report("00_PipelineSummary", pipeline_summary)

# 保存报告
_save_report()


# ═══════════════════════════════════════════════════════
# 最终汇总打印
# ═══════════════════════════════════════════════════════
sep = "=" * 60
print(f"\n{sep}")
print("  第一步数据清洗完成 —— 汇总统计")
print(sep)
print(f"  {'原始样本量':<20}: {n_raw:>8,}")
print(f"  {'去重后样本量':<20}: {n_dedup:>8,}  (去除重复 bvid {n_raw - n_dedup} 条)")
print(f"  {'标题缺失剔除数量':<20}: {n_empty_title:>8,}")
print(f"  {'domain 异常剔除数量':<20}: {n_abnormal_domain:>8,}")
print(f"  {'封面缺失数量':<20}: {n_cover_missing:>8,}  (已标记并另存，不含于主分析)")
print(f"  {'最终样本量':<20}: {n_final:>8,}")
print(f"\n  四类 domain 分布：")
for _, row in domain_dist.iterrows():
    print(f"    {row['domain']:<12}: {int(row['样本量']):>6,} 条")
print(f"\n  level 分布：")
for _, row in level_dist.iterrows():
    print(f"    level={row['level']:<6}: {int(row['样本量']):>6,} 条")
print(f"\n  采集日期：{_collection_date.strftime('%Y-%m-%d')}")
print(f"  主数据输出：outputs/data_clean_base.csv")
print(f"  清洗报告：outputs/logs/data_cleaning_report.xlsx")
print(sep + "\n")

logger.info("脚本运行完毕。")
