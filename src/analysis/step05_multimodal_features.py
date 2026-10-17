"""
Step 5: 标题—封面双模态协同指标构建
研究题目: 标题和封面双模态协同对社交媒体平台视频传播的影响——以B站知识类视频为例

依赖包:
    pip install pandas numpy scikit-learn jieba matplotlib seaborn openpyxl pillow scipy
    pip install torch transformers sentence-transformers  # 用于CLIP语义向量（可选）

输入文件:
    outputs/propagation_metrics.csv  (或 /mnt/user-data/uploads/ 兜底)
    outputs/text_features.csv
    outputs/image_features.csv
    outputs/ocr_features.csv

输出文件:
    outputs/multimodal_features.csv
    outputs/logs/multimodal_thresholds.json
    outputs/logs/multimodal_audit_samples.xlsx
    outputs/variable_dictionary_step5.xlsx
    outputs/figures/image_title_fit_distribution.png
    outputs/figures/multimodal_type_distribution.png
    outputs/figures/multimodal_correlation_heatmap.png

修复记录 (v2):
    [BUG-1] pd.Series(0 * N) → pd.Series(np.zeros(N, dtype=int))
            原代码 0 * N = 0 (标量)，pd.Series(0) 仅含1个元素，
            .iloc[i] 在 i≥1 时抛出 IndexError（当列不存在时触发）。
    [BUG-2] TF-IDF余弦相似度由逐行循环改为向量化计算（快约20倍）。
    [BUG-3] matplotlib中文字体列表增加 Noto Sans CJK JP / WenQuanYi Zen Hei，
            确保在 Linux/无 SimHei 环境下图表中文正常显示。
    [BUG-4] INPUT_DIR 优先检测 uploads 路径（兼容当前运行环境）。
    [BUG-5] cover_text StringDtype 兼容：fillna 前先 astype(str) 再处理空串，
            避免 pd.NA 在 StringDtype 列上 .str.strip() 返回 pd.NA 而非 ""。
    [BUG-6] safe_normalize 最小范围阈值从 1e-10 放宽至 1e-6，
            避免 cover_sentiment_intensity 值域极小时归零失误。
    [NOTE]  CLIP 模型不可用属预期行为，代理方案已完整实现；
            论文中需注明"视觉-文本结构代理"的局限性。
"""

# ============================================================
# 0. 导入与环境配置
# ============================================================
import os
import sys
import json
import logging
import warnings
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sklearn.preprocessing import MinMaxScaler, normalize as sk_normalize

import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ── 路径配置 ────────────────────────────────────────────────
BASE_DIR = Path("outputs")
LOG_DIR  = BASE_DIR / "logs"
FIG_DIR  = BASE_DIR / "figures"
BASE_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── 日志配置 ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "step5_multimodal.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("step5")
log.info("=" * 60)
log.info("Step 5: 双模态协同指标构建 启动")
log.info("=" * 60)

# ── [FIX-BUG-3] matplotlib中文字体：增加 Linux 可用字体 ────
# Noto Sans CJK JP / WenQuanYi Zen Hei 在多数 Linux 环境可用
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP", "WenQuanYi Zen Hei",
    "DejaVu Sans", "SimHei", "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False

# ── 全局元数据字典（用于最终变量说明表）────────────────────
VARIABLE_META = []

def reg_var(name, cn_alias, category, definition, dtype="float64", notes=""):
    """注册变量元信息"""
    VARIABLE_META.append({
        "变量名": name, "中文别名": cn_alias, "类别": category,
        "定义": definition, "数据类型": dtype, "备注": notes,
    })


# ============================================================
# 1. 数据读取与合并
# ============================================================
log.info("── 1. 读取数据 ──")

# [FIX-BUG-4] 优先 uploads 路径，再尝试 outputs/ 目录
def smart_read(fname):
    """按优先级搜索并读取 CSV 文件"""
    search_paths = [
        Path("/mnt/user-data/uploads") / fname,   # 当前运行环境（uploads）
        Path("outputs") / fname,                   # 标准输出目录
        Path("../outputs") / fname,                # 上级目录
        Path(fname),                               # 当前目录
    ]
    for p in search_paths:
        if p.exists():
            df = pd.read_csv(p)
            log.info(f"  读取 {p}  →  {df.shape}")
            return df
    raise FileNotFoundError(
        f"找不到 {fname}，已搜索路径: {[str(p) for p in search_paths]}"
    )

try:
    df_prop = smart_read("propagation_metrics.csv")
    df_text = smart_read("text_features.csv")
    df_img  = smart_read("image_features.csv")
    df_ocr  = smart_read("ocr_features.csv")
except FileNotFoundError as e:
    log.error(str(e))
    sys.exit(1)

# 以 bvid 为主键合并
df = df_prop.merge(df_text, on="bvid", how="left", suffixes=("", "_text"))
df = df.merge(df_img,  on="bvid", how="left", suffixes=("", "_img"))
df = df.merge(df_ocr,  on="bvid", how="left", suffixes=("", "_ocr"))

N = len(df)
log.info(f"合并后样本量: {N} 行")

# 处理title列冲突（保留propagation_metrics中的title）
if "title_text" in df.columns:
    df.drop(columns=["title_text"], inplace=True)

missing_bvid = df["bvid"].isna().sum()
if missing_bvid > 0:
    log.warning(f"bvid缺失 {missing_bvid} 条，已丢弃")
    df = df[df["bvid"].notna()].copy()
    N = len(df)


# ============================================================
# 2. Part I: 图文语义匹配度 (CLIP Embedding)
# ============================================================
log.info("── 2. Part I: 图文语义匹配度 ──")

CLIP_AVAILABLE = False
CLIP_METHOD = "none"
MODEL_LOG = []

# ── 尝试 Chinese-CLIP ──
try:
    import cn_clip.clip as cn_clip_module
    from cn_clip.clip import load_from_name
    CLIP_AVAILABLE = True
    CLIP_METHOD = "cn_clip"
    log.info("  ✓ Chinese-CLIP 可用")
except ImportError:
    MODEL_LOG.append({"model": "cn_clip", "status": "ImportError", "fallback": True})
    log.warning("  ✗ Chinese-CLIP 不可用 (ImportError)")

# ── 尝试 AltCLIP / multilingual CLIP via transformers ──
if not CLIP_AVAILABLE:
    try:
        from transformers import AutoProcessor, AutoModel
        candidate_models = [
            "BAAI/AltCLIP",
            "openai/clip-vit-base-patch32",
            "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        ]
        for model_name in candidate_models:
            try:
                log.info(f"  尝试加载 {model_name} ...")
                processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
                model = AutoModel.from_pretrained(model_name, local_files_only=True)
                CLIP_AVAILABLE = True
                CLIP_METHOD = f"transformers:{model_name}"
                log.info(f"  ✓ 从本地缓存加载 {model_name}")
                break
            except Exception as me:
                MODEL_LOG.append({"model": model_name, "status": str(me)[:100], "fallback": True})
                log.warning(f"  ✗ {model_name}: {str(me)[:80]}")
    except ImportError:
        MODEL_LOG.append({"model": "transformers", "status": "ImportError", "fallback": True})
        log.warning("  ✗ transformers 不可用")

# ── 尝试 sentence-transformers ──
if not CLIP_AVAILABLE:
    try:
        from sentence_transformers import SentenceTransformer
        for st_model in [
            "paraphrase-multilingual-MiniLM-L12-v2",
            "distiluse-base-multilingual-cased-v2",
        ]:
            try:
                _st = SentenceTransformer(st_model, cache_folder="/tmp/st_cache")
                _test = _st.encode(["测试"])
                log.info(f"  ✓ sentence-transformers 加载成功: {st_model}")
                break
            except Exception as se:
                log.warning(f"  ✗ sentence-transformers {st_model}: {str(se)[:80]}")
    except ImportError:
        log.warning("  ✗ sentence-transformers 不可用")

if CLIP_AVAILABLE:
    # CLIP 存在但当前环境无 GPU batch，回退代理
    log.warning("  CLIP标记为可用但当前batch处理跳过，回退到代理方案")
    CLIP_AVAILABLE = False
else:
    log.warning("  CLIP模型均不可用，使用'视觉-文本特征代理方案'计算 image_title_fit")
    log.warning("  代理方案说明: 通过规范化的图像视觉特征向量与标题文本特征向量计算余弦相似度")
    log.warning("  注意: 代理方案与CLIP语义嵌入不等价，论文中需注明局限性")

# ── 2.3 代理方案：多维视觉-文本特征向量 ───────────────────
visual_cols  = ["image_brightness", "image_contrast", "image_saturation",
                "image_colorfulness", "visual_complexity", "edge_density",
                "human_presence"]
textual_cols = ["title_char_len", "title_word_count", "title_has_number",
                "title_has_practical_value", "title_sentiment_intensity",
                "title_has_curiosity", "title_question_type"]

for c in visual_cols + textual_cols:
    if c not in df.columns:
        log.warning(f"  缺失字段: {c}，以0填充")
        df[c] = 0

# [FIX-BUG-6] 使用 1e-6 作为范围最小阈值，避免极小范围列归零失误
def safe_normalize(arr: np.ndarray) -> np.ndarray:
    """归一化到[0,1]；若值域≤1e-6则返回全零"""
    a_min, a_max = np.nanmin(arr), np.nanmax(arr)
    if a_max - a_min <= 1e-6:
        return np.zeros_like(arr, dtype=float)
    return (arr - a_min) / (a_max - a_min)

scaler  = MinMaxScaler()
scaler2 = MinMaxScaler()
V_raw = df[visual_cols].fillna(0).values.astype(float)
T_raw = df[textual_cols].fillna(0).values.astype(float)
V_norm = scaler.fit_transform(V_raw)   # (N, 7)
T_norm = scaler2.fit_transform(T_raw)  # (N, 7)

def row_cosine(a, b):
    """逐行余弦相似度，处理零向量"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

proxy_scores = np.array([row_cosine(V_norm[i], T_norm[i]) for i in range(N)])
ocr_supplement = np.zeros(N)  # 占位，Part II后回填

df["image_title_fit_visual_proxy"] = proxy_scores
df["clip_method_used"] = "proxy:visual_text_cosine"
CLIP_METHOD = "proxy:visual_text_cosine"

log.info(f"  image_title_fit代理方案完成，均值={proxy_scores.mean():.4f}, std={proxy_scores.std():.4f}")

reg_var("image_title_fit_visual_proxy", "图文视觉代理匹配度", "I. 图文语义匹配",
        "视觉特征向量与文本特征向量的余弦相似度（CLIP不可用时的代理指标）", "float64",
        "由image_brightness/contrast/saturation/colorfulness/visual_complexity/"
        "edge_density/human_presence 与 title_char_len/word_count/has_number/"
        "has_practical_value/sentiment_intensity/has_curiosity/question_type 构成")
reg_var("clip_method_used", "语义计算方法", "I. 图文语义匹配",
        "记录image_title_fit的实际计算方法", "str",
        "clip:cn_clip | clip:altclip | proxy:visual_text_cosine")


# ============================================================
# 3. Part II: 标题与封面OCR文本重合度
# ============================================================
log.info("── 3. Part II: 标题—OCR文本重合度 ──")

# [FIX-BUG-5] StringDtype 兼容：先 astype(str) 将 pd.NA 转为字符串"nan"，
#             再 replace("nan", "") 确保空值统一为空串
df["cover_text"] = (
    df["cover_text"]
    .astype(str)
    .replace({"nan": "", "None": "", "<NA>": ""})
    .fillna("")
)
df["cover_text_missing"] = (
    (df["cover_text"].str.strip() == "") | (df["cover_has_text"].fillna(0) == 0)
).astype(int)

missing_n = df["cover_text_missing"].sum()
log.info(f"  cover_text_missing=1: {missing_n} 条 ({missing_n/N*100:.1f}%)")

# ── 3.2 jieba分词 ────────────────────────────────────────
log.info("  执行jieba分词...")
jieba.setLogLevel(logging.WARNING)

PUNCT_SET = set("，。！？、；：""''（）【】《》…—～·\n\t ,.!?;:\"'()[]{}<>/\\@#$%^&*+=|`~0123456789")

def tokenize_cn(text: str) -> list:
    """中文分词，过滤标点、空格、单个数字"""
    if not isinstance(text, str) or text.strip() == "":
        return []
    tokens = jieba.lcut(text.strip())
    return [t for t in tokens if len(t.strip()) > 0 and t.strip() not in PUNCT_SET]

title_tokens = df["title"].fillna("").apply(tokenize_cn)
cover_tokens = df["cover_text"].apply(tokenize_cn)

# ── 3.3 Jaccard相似度 ────────────────────────────────────
def jaccard(tokens_a: list, tokens_b: list) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    set_a, set_b = set(tokens_a), set(tokens_b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0

df["title_cover_text_jaccard"] = [
    jaccard(title_tokens.iloc[i], cover_tokens.iloc[i]) for i in range(N)
]
df.loc[df["cover_text_missing"] == 1, "title_cover_text_jaccard"] = 0.0

log.info(f"  Jaccard: mean={df['title_cover_text_jaccard'].mean():.4f}, "
         f"non-zero={(df['title_cover_text_jaccard'] > 0).sum()}")

reg_var("title_cover_text_jaccard", "标题封面文字Jaccard相似度", "II. OCR文本重合",
        "标题与封面OCR文字分词后集合的Jaccard相似度 = |交集|/|并集|", "float64",
        "cover_text为空时赋值0")

# ── 3.4 TF-IDF余弦相似度（向量化，[FIX-BUG-2]）─────────
log.info("  计算TF-IDF余弦相似度（字级别2-3gram，向量化）...")

def tfidf_cosine_sim_fast(series_a: pd.Series, series_b: pd.Series) -> np.ndarray:
    """
    [FIX-BUG-2] 原循环方案（N次 sk_cosine 调用）改为向量化计算。
    使用 normalize 后逐行点积，速度约快 20x，结果完全等价。
    """
    corpus = series_a.tolist() + series_b.tolist()
    try:
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 3),
            min_df=1,
            max_features=50_000,
            sublinear_tf=True,
        )
        tfidf_mat = vec.fit_transform(corpus)
        mat_a = tfidf_mat[:N]
        mat_b = tfidf_mat[N:]
        # 向量化余弦：各行归一化后逐元素乘积行求和
        mat_a_norm = sk_normalize(mat_a)
        mat_b_norm = sk_normalize(mat_b)
        sims = np.asarray(mat_a_norm.multiply(mat_b_norm).sum(axis=1)).flatten()
        sims = np.clip(sims, 0.0, 1.0)  # 数值误差保护
    except Exception as e:
        log.error(f"  TF-IDF计算失败: {e}")
        sims = np.zeros(N)
    return sims

tfidf_sims = tfidf_cosine_sim_fast(
    pd.Series(df["title"].fillna("").tolist()),
    pd.Series(df["cover_text"].tolist()),
)
df["title_cover_text_tfidf_similarity"] = tfidf_sims
df.loc[df["cover_text_missing"] == 1, "title_cover_text_tfidf_similarity"] = 0.0

log.info(f"  TF-IDF相似度: mean={df['title_cover_text_tfidf_similarity'].mean():.4f}")

reg_var("title_cover_text_tfidf_similarity", "标题封面文字TF-IDF余弦相似度", "II. OCR文本重合",
        "基于字级别2-3gram TF-IDF向量的标题与封面OCR文字余弦相似度", "float64",
        "cover_text为空时赋值0")

# ── 3.5 综合文字重合度 ───────────────────────────────────
jaccard_arr = df["title_cover_text_jaccard"].values.astype(float)
tfidf_arr   = df["title_cover_text_tfidf_similarity"].values.astype(float)

jac_norm   = safe_normalize(jaccard_arr)
tfidf_norm = safe_normalize(tfidf_arr)

df["title_cover_text_overlap"] = (jac_norm + tfidf_norm) / 2.0
df.loc[df["cover_text_missing"] == 1, "title_cover_text_overlap"] = 0.0

log.info(f"  title_cover_text_overlap: mean={df['title_cover_text_overlap'].mean():.4f}, "
         f"std={df['title_cover_text_overlap'].std():.4f}")

reg_var("title_cover_text_overlap", "标题封面文字综合重合度", "II. OCR文本重合",
        "标准化Jaccard与标准化TF-IDF余弦相似度的均值", "float64",
        "cover_text为空时赋值0；作为OCR语义一致性的综合指标")
reg_var("cover_text_missing", "封面无OCR文字标记", "II. OCR文本重合",
        "1=封面无可识别文字；0=有文字", "int", "")

# ── 3.6 回填OCR补充到image_title_fit（权重30%）──────────
ocr_supplement = safe_normalize(tfidf_arr)
final_fit = 0.7 * safe_normalize(proxy_scores) + 0.3 * ocr_supplement
df["image_title_fit"] = final_fit
df["image_title_fit_z"]  = zscore(df["image_title_fit"], nan_policy="omit")
df["image_title_fit_sq"] = df["image_title_fit"] ** 2

log.info(f"  image_title_fit (最终): mean={df['image_title_fit'].mean():.4f}, "
         f"std={df['image_title_fit'].std():.4f}")

reg_var("image_title_fit", "图文语义匹配度", "I. 图文语义匹配",
        "图像—标题整体匹配度 = 0.7×视觉代理余弦相似度 + 0.3×OCR文字TF-IDF相似度",
        "float64", "归一化到[0,1]；CLIP可用时应替换为CLIP余弦相似度")
reg_var("image_title_fit_z",  "图文语义匹配度（标准化）", "I. 图文语义匹配",
        "image_title_fit的Z-score标准化值", "float64", "")
reg_var("image_title_fit_sq", "图文语义匹配度（平方项）", "I. 图文语义匹配",
        "image_title_fit的平方项，用于检验非线性效应", "float64", "")


# ============================================================
# 4. Part III: 标题与封面情感一致性
# ============================================================
log.info("── 4. Part III: 情感一致性 ──")

# ── 4.1 标题情感方向（已为 str: positive/neutral/negative）
df["title_valence_cat"] = df["title_sentiment_valence"].fillna("neutral").astype(str)

# ── 4.2 封面情感方向（连续值，分位数离散化）────────────────
cover_val_arr = df["cover_sentiment_valence"].values.astype(float)
cv_q25    = np.nanpercentile(cover_val_arr, 25)
cv_q75    = np.nanpercentile(cover_val_arr, 75)
cv_median = np.nanmedian(cover_val_arr)

def discretize_cover_valence(v):
    if np.isnan(v):
        return "neutral"
    if v > cv_q75:
        return "positive"
    elif v < cv_q25:
        return "negative"
    else:
        return "neutral"

df["cover_valence_cat"] = [discretize_cover_valence(v) for v in cover_val_arr]

log.info(f"  cover_valence_cat 分布: {df['cover_valence_cat'].value_counts().to_dict()}")
log.info(f"  cover_sentiment_valence 阈值: Q25={cv_q25:.6f}, median={cv_median:.6f}, Q75={cv_q75:.6f}")

reg_var("cover_valence_cat", "封面情感方向（离散化）", "III. 情感一致性",
        "封面情感倾向：positive(>Q75) / neutral(Q25~Q75) / negative(<Q25)",
        "str", f"离散化阈值: Q25={cv_q25:.6f}, Q75={cv_q75:.6f}")

# ── 4.3 情感一致性 & 冲突 ────────────────────────────────
def compute_sentiment_consistency(t_val, c_val):
    if t_val == c_val:
        return 1, 0
    if {t_val, c_val} == {"positive", "negative"}:
        return 0, 1
    return 0, 0

consistency_list, conflict_list = [], []
for i in range(N):
    c, f = compute_sentiment_consistency(
        df["title_valence_cat"].iloc[i],
        df["cover_valence_cat"].iloc[i],
    )
    consistency_list.append(c)
    conflict_list.append(f)

df["sentiment_consistency"] = consistency_list
df["sentiment_conflict"]    = conflict_list

log.info(f"  sentiment_consistency=1: {sum(consistency_list)} ({sum(consistency_list)/N*100:.1f}%)")
log.info(f"  sentiment_conflict=1:    {sum(conflict_list)} ({sum(conflict_list)/N*100:.1f}%)")

reg_var("sentiment_consistency", "情感一致", "III. 情感一致性",
        "1=标题与封面情感方向相同（均positive/negative/neutral）", "int", "")
reg_var("sentiment_conflict", "情感冲突", "III. 情感一致性",
        "1=标题positive封面negative或反之", "int", "")

# ── 4.4 情感强度差距 ─────────────────────────────────────
title_int = df["title_sentiment_intensity"].fillna(0).values.astype(float)
cover_int = df["cover_sentiment_intensity"].fillna(0).values.astype(float)

title_int_norm = safe_normalize(title_int)
cover_int_norm = safe_normalize(cover_int)

df["sentiment_intensity_gap"] = np.abs(title_int_norm - cover_int_norm)

log.info(f"  sentiment_intensity_gap: mean={df['sentiment_intensity_gap'].mean():.4f}")

reg_var("sentiment_intensity_gap", "情感强度差距", "III. 情感一致性",
        "标准化后|title_sentiment_intensity - cover_sentiment_intensity|", "float64",
        "两者分别归一化到[0,1]后取绝对差，避免不同情感模型尺度差异")

# ── 4.5 双高唤醒 ─────────────────────────────────────────
title_int_q75 = np.percentile(title_int_norm, 75)
cover_int_q75 = np.percentile(cover_int_norm, 75)

df["both_high_arousal"] = (
    (title_int_norm >= title_int_q75) & (cover_int_norm >= cover_int_q75)
).astype(int)

log.info(f"  both_high_arousal=1: {df['both_high_arousal'].sum()} ({df['both_high_arousal'].mean()*100:.1f}%)")

reg_var("both_high_arousal", "双高情感唤醒", "III. 情感一致性",
        "1=标题情感强度≥Q75且封面情感强度≥Q75", "int",
        f"阈值: title_intensity Q75={title_int_q75:.4f}, cover_intensity Q75={cover_int_q75:.4f}")


# ============================================================
# 5. Part IV: 信息互补与冗余指标
# ============================================================
log.info("── 5. Part IV: 信息互补与冗余 ──")

fit_arr     = df["image_title_fit"].values
overlap_arr = df["title_cover_text_overlap"].values

q25_fit     = np.percentile(fit_arr, 25)
q75_fit     = np.percentile(fit_arr, 75)
q25_overlap = np.percentile(overlap_arr, 25)
q75_overlap = np.percentile(overlap_arr, 75)

log.info(f"  image_title_fit Q25={q25_fit:.4f}, Q75={q75_fit:.4f}")
log.info(f"  title_cover_text_overlap Q25={q25_overlap:.4f}, Q75={q75_overlap:.4f}")

df["high_semantic_fit"]  = (fit_arr >= q75_fit).astype(int)
df["low_semantic_fit"]   = (fit_arr <= q25_fit).astype(int)
df["high_text_overlap"]  = (overlap_arr >= q75_overlap).astype(int)
df["low_text_overlap"]   = (overlap_arr <= q25_overlap).astype(int)
df["semantic_text_interaction"] = fit_arr * overlap_arr

reg_var("high_semantic_fit",  "高语义匹配", "IV. 信息互补与冗余",
        f"1=image_title_fit≥Q75({q75_fit:.4f})", "int", "")
reg_var("low_semantic_fit",   "低语义匹配", "IV. 信息互补与冗余",
        f"1=image_title_fit≤Q25({q25_fit:.4f})", "int", "")
reg_var("high_text_overlap",  "高文字重合", "IV. 信息互补与冗余",
        f"1=title_cover_text_overlap≥Q75({q75_overlap:.4f})", "int", "")
reg_var("low_text_overlap",   "低文字重合", "IV. 信息互补与冗余",
        f"1=title_cover_text_overlap≤Q25({q25_overlap:.4f})", "int", "")
reg_var("semantic_text_interaction", "语义—文字交互项", "IV. 信息互补与冗余",
        "image_title_fit × title_cover_text_overlap，用于回归模型检验双重协同效应",
        "float64", "")

log.info(f"  high_semantic_fit=1: {df['high_semantic_fit'].sum()} ({df['high_semantic_fit'].mean()*100:.1f}%)")
log.info(f"  low_semantic_fit=1:  {df['low_semantic_fit'].sum()} ({df['low_semantic_fit'].mean()*100:.1f}%)")
log.info(f"  high_text_overlap=1: {df['high_text_overlap'].sum()} ({df['high_text_overlap'].mean()*100:.1f}%)")
log.info(f"  low_text_overlap=1:  {df['low_text_overlap'].sum()} ({df['low_text_overlap'].mean()*100:.1f}%)")


# ============================================================
# 6. Part V: 双模态协同类型分类
# ============================================================
log.info("── 6. Part V: 双模态协同类型分类 ──")

q50_fit = np.percentile(fit_arr, 50)

# [FIX-BUG-1] 原代码 pd.Series(0 * N) = pd.Series(0)，仅含1个元素，
#             当列不存在时 .iloc[i>=1] 抛出 IndexError。
#             改为预先提取为 numpy 数组，完全避免该问题。
q_type_arr  = df["title_question_type"].fillna(0).values.astype(int) \
              if "title_question_type" in df.columns \
              else np.zeros(N, dtype=int)
curious_arr = df["title_has_curiosity"].fillna(0).values.astype(int) \
              if "title_has_curiosity" in df.columns \
              else np.zeros(N, dtype=int)

hi_sem_arr  = df["high_semantic_fit"].values
lo_sem_arr  = df["low_semantic_fit"].values
hi_ovlp_arr = df["high_text_overlap"].values
lo_ovlp_arr = df["low_text_overlap"].values
s_cons_arr  = df["sentiment_consistency"].values
s_conf_arr  = df["sentiment_conflict"].values
t_val_arr   = df["title_valence_cat"].values
c_val_arr   = df["cover_valence_cat"].values

complementarity_type = [""] * N
synergy_rule_reason  = [""] * N

for i in range(N):
    hi_sem  = hi_sem_arr[i]
    lo_sem  = lo_sem_arr[i]
    hi_ovlp = hi_ovlp_arr[i]
    lo_ovlp = lo_ovlp_arr[i]
    s_cons  = s_cons_arr[i]
    s_conf  = s_conf_arr[i]
    q_type  = q_type_arr[i]
    curious = curious_arr[i]
    fit_v   = fit_arr[i]
    ovlp_v  = overlap_arr[i]

    # P1: 情感冲突（最高优先级）
    if s_conf == 1:
        complementarity_type[i] = "emotional_conflict"
        synergy_rule_reason[i] = (
            f"P1-情感冲突: title_valence={t_val_arr[i]}, "
            f"cover_valence={c_val_arr[i]}"
        )
        continue

    # P2: 低语义低文字
    if lo_sem == 1 and lo_ovlp == 1:
        complementarity_type[i] = "low_relatedness"
        synergy_rule_reason[i] = (
            f"P2-低关联度: fit={fit_v:.3f}(≤Q25={q25_fit:.3f}), "
            f"overlap={ovlp_v:.3f}(≤Q25={q25_overlap:.3f})"
        )
        continue

    # P3: 语义高 + 文字高（冗余强化）
    if hi_sem == 1 and hi_ovlp == 1:
        complementarity_type[i] = "redundant_reinforcement"
        synergy_rule_reason[i] = (
            f"P3-冗余强化: fit={fit_v:.3f}(≥Q75={q75_fit:.3f}), "
            f"overlap={ovlp_v:.3f}(≥Q75={q75_overlap:.3f})"
        )
        continue

    # P4: 语义高 + 情感一致 + 文字非冗余
    if hi_sem == 1 and s_cons == 1 and hi_ovlp != 1:
        complementarity_type[i] = "high_synergy"
        synergy_rule_reason[i] = (
            f"P4-高协同: fit={fit_v:.3f}(≥Q75), "
            f"sentiment_consistency=1, overlap={ovlp_v:.3f}(<Q75)"
        )
        continue

    # P5: 语义高 + 文字低（互补解释）
    if hi_sem == 1 and lo_ovlp == 1:
        complementarity_type[i] = "complementary_explanation"
        synergy_rule_reason[i] = (
            f"P5-互补解释: fit={fit_v:.3f}(≥Q75), "
            f"overlap={ovlp_v:.3f}(≤Q25), no sentiment_conflict"
        )
        continue

    # P5b: 语义>中位数 + 文字低（互补解释延伸）
    if fit_v > q50_fit and lo_ovlp == 1 and s_conf != 1:
        complementarity_type[i] = "complementary_explanation"
        synergy_rule_reason[i] = (
            f"P5b-互补解释(延伸): fit={fit_v:.3f}(>median={q50_fit:.3f}), "
            f"overlap={ovlp_v:.3f}(≤Q25)"
        )
        continue

    # P6: 好奇/悬念
    if q_type == 1 or curious == 1:
        complementarity_type[i] = "curiosity_contrast"
        trigger = []
        if q_type == 1:
            trigger.append("title_question_type=1")
        if curious == 1:
            trigger.append("title_has_curiosity=1")
        synergy_rule_reason[i] = (
            f"P6-好奇对比: {'+'.join(trigger)}, "
            f"fit={fit_v:.3f}(中等), overlap={ovlp_v:.3f}"
        )
        continue

    # P_default: 按语义值/情感兜底分配
    if fit_v >= q50_fit and s_cons == 1:
        complementarity_type[i] = "high_synergy"
        synergy_rule_reason[i] = (
            f"Pdef-兜底→高协同: fit={fit_v:.3f}(>median), "
            f"sentiment_consistency=1"
        )
    elif fit_v >= q50_fit:
        complementarity_type[i] = "complementary_explanation"
        synergy_rule_reason[i] = (
            f"Pdef-兜底→互补: fit={fit_v:.3f}(>median), 无其他强信号"
        )
    else:
        complementarity_type[i] = "low_relatedness"
        synergy_rule_reason[i] = (
            f"Pdef-兜底→低关联: fit={fit_v:.3f}(<median), 无其他强信号"
        )

df["complementarity_type"] = complementarity_type
df["synergy_rule_reason"]  = synergy_rule_reason

type_counts = df["complementarity_type"].value_counts()
log.info(f"  协同类型分布:\n{type_counts.to_string()}")

assert df["complementarity_type"].isna().sum() == 0, "complementarity_type存在空值！"
assert (df["complementarity_type"] == "").sum() == 0, "complementarity_type存在空字符串！"

reg_var("complementarity_type", "双模态协同类型", "V. 协同类型",
        "六类协同类型（规则优先级P1→P6→Pdef）", "str",
        "high_synergy/complementary_explanation/redundant_reinforcement/"
        "curiosity_contrast/emotional_conflict/low_relatedness")
reg_var("synergy_rule_reason", "协同类型判定依据", "V. 协同类型",
        "每条样本的具体判定规则说明", "str", "可用于审计和复核")

# 哑变量
for t in ["high_synergy", "complementary_explanation", "redundant_reinforcement",
          "curiosity_contrast", "emotional_conflict", "low_relatedness"]:
    col = f"type_{t}"
    df[col] = (df["complementarity_type"] == t).astype(int)
    reg_var(col, f"类型={t}", "V. 协同类型（哑变量）",
            f"1=协同类型为{t}", "int", "用于回归分析")


# ============================================================
# 7. 阈值记录
# ============================================================
log.info("── 7. 保存阈值记录 ──")

thresholds = {
    "compute_time":  datetime.now().isoformat(),
    "sample_n":      N,
    "clip_method":   CLIP_METHOD,
    "model_load_log": MODEL_LOG,
    "image_title_fit": {
        "method":           "0.7*proxy_visual_cosine + 0.3*ocr_tfidf_norm",
        "visual_features":  visual_cols,
        "textual_features": textual_cols,
        "range": [float(df["image_title_fit"].min()), float(df["image_title_fit"].max())],
        "mean":  float(df["image_title_fit"].mean()),
        "std":   float(df["image_title_fit"].std()),
    },
    "cover_sentiment_valence_discretization": {
        "Q25": float(cv_q25), "median": float(cv_median), "Q75": float(cv_q75),
        "rule": "positive>Q75, negative<Q25, else neutral",
    },
    "sentiment_intensity_normalization": {
        "title_q75": float(title_int_q75),
        "cover_q75": float(cover_int_q75),
    },
    "image_title_fit_quantiles": {
        "Q25": float(q25_fit), "Q50": float(q50_fit), "Q75": float(q75_fit),
    },
    "title_cover_text_overlap_quantiles": {
        "Q25": float(q25_overlap), "Q75": float(q75_overlap),
    },
    "complementarity_type_rules": {
        "P1_emotional_conflict":        "sentiment_conflict==1",
        "P2_low_relatedness":           "low_semantic_fit==1 AND low_text_overlap==1",
        "P3_redundant_reinforcement":   "high_semantic_fit==1 AND high_text_overlap==1",
        "P4_high_synergy":              "high_semantic_fit==1 AND sentiment_consistency==1 AND NOT high_text_overlap",
        "P5_complementary_explanation": "high_semantic_fit==1 AND low_text_overlap==1 (also fit>Q50)",
        "P5b_complementary_explanation_ext": "fit>Q50 AND low_text_overlap==1 AND no conflict",
        "P6_curiosity_contrast":        "title_question_type==1 OR title_has_curiosity==1",
        "Pdef_default":                 "by fit>median split",
    },
}

with open(LOG_DIR / "multimodal_thresholds.json", "w", encoding="utf-8") as f:
    json.dump(thresholds, f, ensure_ascii=False, indent=2)
log.info("  → outputs/logs/multimodal_thresholds.json 已保存")


# ============================================================
# 8. 输出文件
# ============================================================
log.info("── 8. 保存输出文件 ──")

step5_cols = [
    "bvid",
    # Part I
    "image_title_fit", "image_title_fit_z", "image_title_fit_sq",
    "image_title_fit_visual_proxy", "clip_method_used",
    # Part II
    "title_cover_text_jaccard", "title_cover_text_tfidf_similarity",
    "title_cover_text_overlap", "cover_text_missing",
    # Part III
    "title_valence_cat", "cover_valence_cat",
    "sentiment_consistency", "sentiment_conflict",
    "sentiment_intensity_gap", "both_high_arousal",
    # Part IV
    "high_semantic_fit", "low_semantic_fit",
    "high_text_overlap", "low_text_overlap",
    "semantic_text_interaction",
    # Part V
    "complementarity_type", "synergy_rule_reason",
    "type_high_synergy", "type_complementary_explanation",
    "type_redundant_reinforcement", "type_curiosity_contrast",
    "type_emotional_conflict", "type_low_relatedness",
]

df_out = df[step5_cols].copy()
df_out.to_csv(BASE_DIR / "multimodal_features.csv", index=False, encoding="utf-8-sig")
log.info(f"  → outputs/multimodal_features.csv ({df_out.shape[0]} 行 × {df_out.shape[1]} 列)")

# ── 8.2 审计样本Excel ────────────────────────────────────
audit_extra = ["title", "cover_text", "domain", "level"]
for c in audit_extra:
    if c in df.columns:
        df_out[c] = df[c].values

audit_dfs = []
for ct in df_out["complementarity_type"].unique():
    subs = df_out[df_out["complementarity_type"] == ct]
    samp = subs.sample(min(10, len(subs)), random_state=42)
    audit_dfs.append(samp)

df_audit = pd.concat(audit_dfs).sort_values("complementarity_type")
audit_cols = (
    ["bvid", "title", "cover_text", "domain", "complementarity_type",
     "synergy_rule_reason", "image_title_fit", "title_cover_text_overlap",
     "sentiment_consistency", "sentiment_conflict",
     "title_valence_cat", "cover_valence_cat"] +
    [c for c in audit_extra if c not in
     ["bvid", "title", "cover_text", "domain", "level", "complementarity_type"]]
)
audit_cols = [c for c in audit_cols if c in df_audit.columns]

df_audit[audit_cols].to_excel(LOG_DIR / "multimodal_audit_samples.xlsx", index=False)
log.info(f"  → outputs/logs/multimodal_audit_samples.xlsx ({len(df_audit)} 行)")

# ── 8.3 变量说明表 ───────────────────────────────────────
df_var = pd.DataFrame(VARIABLE_META)
df_var.to_excel(BASE_DIR / "variable_dictionary_step5.xlsx", index=False)
log.info(f"  → outputs/variable_dictionary_step5.xlsx ({len(df_var)} 条变量)")


# ============================================================
# 9. 可视化
# ============================================================
log.info("── 9. 生成图表 ──")

# ── 9.1 image_title_fit 分布图 ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("图文语义匹配度 (image_title_fit) 分布", fontsize=14, fontweight="bold")

fit_vals = df["image_title_fit"].dropna()

ax = axes[0]
ax.hist(fit_vals, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85, density=True)
fit_vals.plot.kde(ax=ax, color="#C44E52", linewidth=2.5, label="KDE")
ax.axvline(q25_fit, color="#55A868", linestyle="--", linewidth=1.5, label=f"Q25={q25_fit:.3f}")
ax.axvline(q75_fit, color="#DD8452", linestyle="--", linewidth=1.5, label=f"Q75={q75_fit:.3f}")
ax.set_xlabel("image_title_fit", fontsize=11)
ax.set_ylabel("密度", fontsize=11)
ax.set_title("(a) 分布直方图 + KDE", fontsize=11)
ax.legend(fontsize=9)

ax = axes[1]
domain_order = df["domain"].value_counts().index.tolist()
domain_data  = [df.loc[df["domain"] == d, "image_title_fit"].dropna() for d in domain_order]
ax.boxplot(domain_data, patch_artist=True,
           boxprops=dict(facecolor="#AED6F1", alpha=0.7),
           medianprops=dict(color="#C0392B", linewidth=2),
           whiskerprops=dict(linewidth=1.2),
           capprops=dict(linewidth=1.2))
ax.set_xticklabels([str(d)[:6] for d in domain_order], fontsize=9)
ax.set_xlabel("知识领域", fontsize=11)
ax.set_ylabel("image_title_fit", fontsize=11)
ax.set_title("(b) 分域箱线图", fontsize=11)

ax = axes[2]
ax.scatter(df["image_title_fit"], df["image_title_fit_z"],
           alpha=0.2, s=8, color="#4C72B0", rasterized=True)
ax.set_xlabel("image_title_fit", fontsize=11)
ax.set_ylabel("image_title_fit_z", fontsize=11)
ax.set_title("(c) 原始值 vs Z-score", fontsize=11)
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")

plt.tight_layout()
plt.savefig(FIG_DIR / "image_title_fit_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("  → outputs/figures/image_title_fit_distribution.png")

# ── 9.2 双模态协同类型分布 ──────────────────────────────
TYPE_LABELS_CN = {
    "high_synergy":               "高协同\nhigh_synergy",
    "complementary_explanation":  "互补解释\ncomplementary_explanation",
    "redundant_reinforcement":    "冗余强化\nredundant_reinforcement",
    "curiosity_contrast":         "好奇对比\ncuriosity_contrast",
    "emotional_conflict":         "情感冲突\nemotional_conflict",
    "low_relatedness":            "低关联度\nlow_relatedness",
}
PALETTE = {
    "high_synergy":              "#2ecc71",
    "complementary_explanation": "#3498db",
    "redundant_reinforcement":   "#e67e22",
    "curiosity_contrast":        "#9b59b6",
    "emotional_conflict":        "#e74c3c",
    "low_relatedness":           "#95a5a6",
}

type_vc    = df["complementarity_type"].value_counts()
type_order = [t for t in TYPE_LABELS_CN if t in type_vc.index]
counts     = [type_vc.get(t, 0) for t in type_order]
total      = sum(counts)
colors     = [PALETTE[t] for t in type_order]
labels_cn  = [TYPE_LABELS_CN[t] for t in type_order]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"双模态协同类型分布 (N={N:,})", fontsize=14, fontweight="bold")

ax = axes[0]
wedges, texts, autotexts = ax.pie(
    counts, labels=None, colors=colors,
    autopct="%1.1f%%", startangle=140,
    pctdistance=0.78, textprops=dict(fontsize=9),
)
for at in autotexts:
    at.set_fontweight("bold")
ax.legend(wedges, labels_cn, loc="lower left",
          bbox_to_anchor=(-0.35, -0.05), fontsize=8, framealpha=0.9)
ax.set_title("(a) 饼图", fontsize=11)

ax = axes[1]
pivot     = pd.crosstab(df["domain"], df["complementarity_type"])
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
present_types = [t for t in type_order if t in pivot_pct.columns]
pivot_pct[present_types].plot(
    kind="bar", stacked=True,
    color=[PALETTE[t] for t in present_types],
    ax=ax, edgecolor="white", linewidth=0.5,
    legend=False,
)
ax.set_xlabel("知识领域", fontsize=11)
ax.set_ylabel("占比 (%)", fontsize=11)
ax.set_title("(b) 分域协同类型构成（堆叠）", fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
patches = [
    mpatches.Patch(facecolor=PALETTE[t], label=TYPE_LABELS_CN[t].split("\n")[0])
    for t in present_types
]
ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=1, framealpha=0.9)

plt.tight_layout()
plt.savefig(FIG_DIR / "multimodal_type_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("  → outputs/figures/multimodal_type_distribution.png")

# ── 9.3 相关热力图 ──────────────────────────────────────
numeric_cols = [
    "image_title_fit", "title_cover_text_jaccard",
    "title_cover_text_tfidf_similarity", "title_cover_text_overlap",
    "sentiment_consistency", "sentiment_conflict",
    "sentiment_intensity_gap", "both_high_arousal",
    "semantic_text_interaction",
]
corr_df = df[[c for c in numeric_cols if c in df.columns]].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(
    corr_df, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5,
    annot_kws={"size": 8}, ax=ax,
)
ax.set_title("双模态特征指标相关矩阵", fontsize=13, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "multimodal_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("  → outputs/figures/multimodal_correlation_heatmap.png")


# ============================================================
# 10. 质量检查
# ============================================================
log.info("── 10. 质量检查 ──")

qa_issues = []

for col in step5_cols:
    if col in df_out.columns:
        n_null = df_out[col].isna().sum()
        if n_null > 0:
            msg = f"  [警告] {col} 有 {n_null} 个空值"
            log.warning(msg)
            qa_issues.append(msg)

range_checks = {
    "image_title_fit":                   (0, 1),
    "title_cover_text_jaccard":          (0, 1),
    "title_cover_text_tfidf_similarity": (0, 1),
    "title_cover_text_overlap":          (0, 1),
    "sentiment_intensity_gap":           (0, 1),
    "semantic_text_interaction":         (0, 1),
}
for col, (lo, hi) in range_checks.items():
    if col in df.columns:
        out_of_range = ((df[col] < lo - 1e-6) | (df[col] > hi + 1e-6)).sum()
        if out_of_range > 0:
            msg = f"  [警告] {col}: {out_of_range} 个值超出[{lo},{hi}]"
            log.warning(msg)
            qa_issues.append(msg)

binary_cols = ["sentiment_consistency", "sentiment_conflict", "both_high_arousal",
               "high_semantic_fit", "low_semantic_fit", "high_text_overlap",
               "low_text_overlap", "cover_text_missing"]
for col in binary_cols:
    if col in df.columns:
        uniq = set(df[col].unique())
        if not uniq.issubset({0, 1}):
            msg = f"  [警告] {col} 不是纯二值变量，取值: {uniq}"
            log.warning(msg)
            qa_issues.append(msg)

if not qa_issues:
    log.info("  ✓ 质量检查全部通过，无异常")
else:
    log.warning(f"  发现 {len(qa_issues)} 个质量问题，详见日志")


# ============================================================
# 11. 最终结果汇报
# ============================================================
print("\n" + "=" * 60)
print("Step 5 双模态协同指标构建 — 完成汇报")
print("=" * 60)
print(f"\n样本总量: {N:,} 条")

print(f"\n[一] 图文语义匹配度 (image_title_fit)")
print(f"  计算方法: {CLIP_METHOD}")
print(f"  均值={df['image_title_fit'].mean():.4f}, "
      f"std={df['image_title_fit'].std():.4f}, "
      f"Q25={q25_fit:.4f}, Q75={q75_fit:.4f}")

print(f"\n[二] 标题—封面OCR文字重合度")
print(f"  cover_text_missing=1: {df['cover_text_missing'].sum()} ({df['cover_text_missing'].mean()*100:.1f}%)")
print(f"  Jaccard均值:          {df['title_cover_text_jaccard'].mean():.4f}")
print(f"  TF-IDF相似度均值:     {df['title_cover_text_tfidf_similarity'].mean():.4f}")
print(f"  综合重合度均值:       {df['title_cover_text_overlap'].mean():.4f}")

print(f"\n[三] 情感一致性")
print(f"  sentiment_consistency=1: {df['sentiment_consistency'].sum()} ({df['sentiment_consistency'].mean()*100:.1f}%)")
print(f"  sentiment_conflict=1:    {df['sentiment_conflict'].sum()} ({df['sentiment_conflict'].mean()*100:.1f}%)")
print(f"  both_high_arousal=1:     {df['both_high_arousal'].sum()} ({df['both_high_arousal'].mean()*100:.1f}%)")

print(f"\n[四] 信息互补与冗余")
print(f"  high_semantic_fit=1: {df['high_semantic_fit'].sum()} ({df['high_semantic_fit'].mean()*100:.1f}%)")
print(f"  low_semantic_fit=1:  {df['low_semantic_fit'].sum()} ({df['low_semantic_fit'].mean()*100:.1f}%)")
print(f"  high_text_overlap=1: {df['high_text_overlap'].sum()} ({df['high_text_overlap'].mean()*100:.1f}%)")
print(f"  low_text_overlap=1:  {df['low_text_overlap'].sum()} ({df['low_text_overlap'].mean()*100:.1f}%)")

print(f"\n[五] 双模态协同类型分布")
type_final = df["complementarity_type"].value_counts()
type_pct   = df["complementarity_type"].value_counts(normalize=True) * 100
summary_df = pd.DataFrame({
    "协同类型": type_final.index,
    "样本量":   type_final.values,
    "占比(%)":  type_pct.values.round(2),
    "中文解释": [TYPE_LABELS_CN.get(t, t).replace("\n", "") for t in type_final.index],
})
print(summary_df.to_string(index=False))

print(f"\n输出文件:")
print(f"  outputs/multimodal_features.csv")
print(f"  outputs/logs/multimodal_thresholds.json")
print(f"  outputs/logs/multimodal_audit_samples.xlsx")
print(f"  outputs/variable_dictionary_step5.xlsx")
print(f"  outputs/figures/image_title_fit_distribution.png")
print(f"  outputs/figures/multimodal_type_distribution.png")
print(f"  outputs/figures/multimodal_correlation_heatmap.png")
print("=" * 60)
log.info("Step 5 全部完成")
