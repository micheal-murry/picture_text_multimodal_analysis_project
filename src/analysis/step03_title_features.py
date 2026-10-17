"""
=============================================================================
第三步：标题文本特征提取
研究题目：标题和封面双模态协同对社交媒体平台视频传播的影响——以B站知识类视频为例
=============================================================================
依赖包：
    pip install pandas numpy jieba openpyxl matplotlib seaborn scipy
    可选（情感分析，优先尝试）：
        pip install transformers torch
    fallback：使用内置情感词典法

输入文件：
    outputs/propagation_metrics.csv

输出文件：
    outputs/text_features.csv
    outputs/variable_dictionary_step3.xlsx
    outputs/logs/title_feature_audit_samples.xlsx
    outputs/figures/title_length_distribution.png
    outputs/figures/title_curiosity_practical_rate.png
    outputs/logs/step3_title_features.log
=============================================================================
"""

import os
import sys
import re
import math
import logging
import warnings
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. 路径与日志初始化
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(".")
OUT_DIR    = BASE_DIR / "outputs"
FIG_DIR    = OUT_DIR / "figures"
LOG_DIR    = OUT_DIR / "logs"

for d in [OUT_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "step3_title_features.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("第三步：标题文本特征提取 开始运行")
log.info("运行时间：" + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
log.info("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 中文字体配置
# ─────────────────────────────────────────────────────────────────────────────

def setup_chinese_font():
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                prop = fm.FontProperties(fname=p)
                plt.rcParams["font.family"] = prop.get_name()
                log.info("中文字体已加载：" + p)
                return p
            except Exception:
                continue
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    log.warning("未找到中文字体文件，图表中文可能显示为方块；建议安装 fonts-wqy-zenhei。")
    return None

FONT_PATH = setup_chinese_font()
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────────────────────────────────────
# 2. 词典定义
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_WORDS = [
    "为什么", "怎么", "如何", "是否", "吗", "呢", "何以", "到底",
    "怎样", "为何", "何为", "何种", "哪些", "哪个", "几个",
    "什么", "谁", "哪", "多少", "几",
]

CURIOSITY_WORDS = [
    "为什么", "到底", "真相", "秘密", "竟然", "离谱", "你不知道",
    "怎么回事", "首次", "全网", "硬核", "千万", "别再", "原来",
    "揭秘", "反转", "颠覆", "没想到", "隐藏", "冷知识", "误区",
    "内幕", "答案", "万万没想到", "惊了", "震惊", "想不到", "不可思议",
    "意外", "突破", "史上", "最强", "最全", "独家", "罕见", "极少",
    "绝了", "真的假的", "难以置信",
]

PRACTICAL_WORDS = [
    "方法", "教程", "指南", "入门", "避坑", "技巧", "原理",
    "解释", "看懂", "学会", "总结", "复盘", "干货", "经验",
    "攻略", "建议", "清单", "步骤", "框架", "逻辑", "模型",
    "工具", "模板", "教学", "实操", "拆解", "速成", "必看",
    "必学", "秘籍", "宝典", "手册", "笔记", "精华", "要点",
]

CN_NUMBERS = "零一二三四五六七八九十百千万亿"

QUANTIFIERS = [
    "个", "种", "条", "分钟", "小时", "天", "年", "次",
    "招", "步", "点", "款", "期", "集", "节", "件",
    "本", "册", "套", "项", "类", "段", "章", "篇",
]

POSITIVE_WORDS = [
    "好", "棒", "赞", "优秀", "完美", "精彩", "厉害", "强",
    "牛", "爽", "喜欢", "爱", "开心", "快乐", "美好", "幸福",
    "成功", "进步", "提升", "突破", "超越", "胜利",
    "推荐", "必看", "干货", "精华", "神", "绝", "超级",
    "超棒", "超好", "最好", "顶级", "一流", "优质",
]

NEGATIVE_WORDS = [
    "差", "烂", "糟", "坏", "失败", "错误", "问题", "困难",
    "危险", "风险", "损失", "亏", "骗", "假", "坑", "避",
    "错", "不好", "难", "苦", "痛", "悲", "惨", "崩",
    "完蛋", "垃圾", "骗局", "套路", "坑人", "离谱", "血亏",
]

def add_custom_words_to_jieba():
    all_custom = set(CURIOSITY_WORDS + PRACTICAL_WORDS + QUESTION_WORDS)
    for w in all_custom:
        if len(w) >= 2:
            jieba.add_word(w, freq=100000)
    log.info("jieba 自定义词典加载完成，共 " + str(len(all_custom)) + " 个词。")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 情感分析模块
# ─────────────────────────────────────────────────────────────────────────────

SENTIMENT_METHOD = "unknown"

def try_load_roberta():
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        log.info("正在尝试加载中文 RoBERTa 情感模型...")
        tokenizer = AutoTokenizer.from_pretrained(
            "uer/roberta-base-finetuned-jd-binary-chinese",
            local_files_only=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "uer/roberta-base-finetuned-jd-binary-chinese",
            local_files_only=False,
        )
        sentiment_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            truncation=True,
            max_length=64,
        )
        log.info("RoBERTa 情感模型加载成功。")
        return sentiment_pipeline
    except Exception as e:
        log.warning("RoBERTa 模型加载失败（" + str(e) + "），将使用情感词典法作为 fallback。")
        return None


def sentiment_dict_method(title):
    if not title or not isinstance(title, str):
        return 0.5
    pos = sum(1 for w in POSITIVE_WORDS if w in title)
    neg = sum(1 for w in NEGATIVE_WORDS if w in title)
    total = pos + neg
    if total == 0:
        return 0.5
    return round(0.5 + (pos - neg) / (2 * total), 4)


def batch_sentiment_roberta(titles, pipeline_fn):
    results = []
    batch_size = 64
    log.info("RoBERTa 情感分析：共 " + str(len(titles)) + " 条标题。")
    for i in range(0, len(titles), batch_size):
        batch = [str(t)[:64] if t else "" for t in titles[i: i + batch_size]]
        try:
            preds = pipeline_fn(batch)
            for pred in preds:
                label = pred.get("label", "").upper()
                score = pred.get("score", 0.5)
                if "1" in label or "POS" in label:
                    results.append(round(score, 4))
                else:
                    results.append(round(1 - score, 4))
        except Exception as e:
            log.warning("RoBERTa 批次推断失败：" + str(e) + "，该批次使用词典法补充。")
            for t in batch:
                results.append(sentiment_dict_method(t))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. 特征提取函数
# ─────────────────────────────────────────────────────────────────────────────

_NUMBER_PHRASE_PATTERN = re.compile(
    r'(\d+\.?\d*\s*(?:分钟|小时|天|年|次|个|种|条|步|招|款|点|集|节|件|本|套|项|类|章|篇|%|％|倍|层|位|名|人|元|块|万|亿)?)'
    r'|([一二三四五六七八九十百千万亿]+\s*(?:个|种|条|分钟|小时|天|年|次|步|招|款|点|集|节|件|本|套|项|类|章|篇)?)'
)

def extract_number_phrases(title):
    if not title:
        return []
    matches = _NUMBER_PHRASE_PATTERN.findall(title)
    phrases = []
    for m in matches:
        phrase = (m[0] or m[1]).strip()
        if phrase:
            phrases.append(phrase)
    return phrases


_PUNCT_PATTERN = re.compile(
    r'[，。！？、；：""''（）【】《》—……·,.!?;:\'"()\[\]<>]'
)

def extract_title_features(title):
    feats = {}
    if not isinstance(title, str):
        title = str(title) if title else ""

    # --- 一、长度与结构 ---
    feats["title_char_len"]    = len(title)
    feats["title_char_len_sq"] = len(title) ** 2

    words = [w.strip() for w in jieba.lcut(title) if w.strip()]
    feats["title_word_count"] = len(words)
    feats["title_punctuation_count"] = len(_PUNCT_PATTERN.findall(title))

    # --- 二、句式与标点 ---
    feats["title_question_mark"]    = int(bool(re.search(r'[?？]', title)))
    feats["title_exclamation_mark"] = int(bool(re.search(r'[!！]', title)))

    has_q_punct = bool(re.search(r'[?？]', title))
    has_q_word  = any(w in title for w in QUESTION_WORDS)
    feats["title_question_type"]    = int(has_q_punct or has_q_word)
    feats["title_exclamation_type"] = feats["title_exclamation_mark"]

    # --- 三、数字与数量表达 ---
    has_arabic     = bool(re.search(r'\d', title))
    has_chinese_num = any(c in title for c in CN_NUMBERS)
    feats["title_has_number"]  = int(has_arabic or has_chinese_num)

    arabic_nums = re.findall(r'\d+', title)
    cn_num_seqs = re.findall('[' + CN_NUMBERS + ']+', title)
    feats["title_number_count"] = len(arabic_nums) + len(cn_num_seqs)

    feats["title_has_quantifier"] = int(any(q in title for q in QUANTIFIERS))

    phrases = extract_number_phrases(title)
    feats["title_number_phrase"] = "|".join(phrases) if phrases else ""

    # --- 四、悬念与好奇心 ---
    curiosity_hits = [w for w in CURIOSITY_WORDS if w in title]
    feats["curiosity_marker_count"] = len(curiosity_hits)
    feats["title_has_curiosity"]    = int(len(curiosity_hits) > 0)

    # --- 五、知识实用性 ---
    practical_hits = [w for w in PRACTICAL_WORDS if w in title]
    feats["practical_value_word_count"] = len(practical_hits)
    feats["title_has_practical_value"]  = int(len(practical_hits) > 0)

    return feats


def compute_clickbait_score(df):
    w1, w2, w3, w4 = 0.40, 0.25, 0.20, 0.15
    cmax   = df["curiosity_marker_count"].max()
    c_norm = df["curiosity_marker_count"] / cmax if cmax > 0 else df["curiosity_marker_count"] * 0
    if "title_sentiment_intensity" in df.columns:
        si_max  = df["title_sentiment_intensity"].max()
        si_norm = df["title_sentiment_intensity"] / si_max if si_max > 0 else df["title_sentiment_intensity"] * 0
    else:
        si_norm = pd.Series(0, index=df.index)
    raw = (
        w1 * c_norm
        + w2 * df["title_question_type"]
        + w3 * df["title_exclamation_type"]
        + w4 * si_norm
    )
    rmin, rmax = raw.min(), raw.max()
    if rmax > rmin:
        return ((raw - rmin) / (rmax - rmin)).round(4)
    return raw.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    # ── 5.1 读取数据 ──────────────────────────────────────────────────────────
    input_path = OUT_DIR / "propagation_metrics.csv"
    if not input_path.exists():
        alt_path = Path("/mnt/user-data/uploads/propagation_metrics.csv")
        if alt_path.exists():
            import shutil
            shutil.copy(alt_path, input_path)
            log.info("从 uploads 复制输入文件到 " + str(input_path))
        else:
            log.error("输入文件不存在：" + str(input_path))
            sys.exit(1)

    log.info("读取输入文件：" + str(input_path))
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    log.info("原始数据行数：" + str(len(df)) + "，列数：" + str(len(df.columns)))

    for col in ["bvid", "title"]:
        if col not in df.columns:
            log.error("缺少必要列：" + col)
            sys.exit(1)

    n_missing = df["title"].isna().sum()
    if n_missing > 0:
        log.warning("title 字段缺失 " + str(n_missing) + " 条，以空字符串填充。")
        df["title"] = df["title"].fillna("")

    # ── 5.2 初始化 jieba ──────────────────────────────────────────────────────
    log.info("初始化 jieba 分词器...")
    add_custom_words_to_jieba()

    # ── 5.3 规则特征提取 ──────────────────────────────────────────────────────
    log.info("开始提取标题规则特征，共 " + str(len(df)) + " 条...")
    t0 = time.time()
    all_feats = []
    for idx, row in df.iterrows():
        feats = extract_title_features(row["title"])
        feats["bvid"]  = row["bvid"]
        feats["title"] = row["title"]
        all_feats.append(feats)
        if (idx + 1) % 500 == 0:
            log.info("  已处理 " + str(idx + 1) + "/" + str(len(df)) + " 条...")

    feat_df = pd.DataFrame(all_feats)
    log.info("规则特征提取完成，耗时 " + str(round(time.time()-t0, 1)) + "s。")

    # ── 5.4 情感分析 ──────────────────────────────────────────────────────────
    log.info("开始情感分析...")
    sentiment_pipeline = try_load_roberta()
    global SENTIMENT_METHOD

    if sentiment_pipeline is not None:
        SENTIMENT_METHOD = "RoBERTa (uer/roberta-base-finetuned-jd-binary-chinese)"
        scores = batch_sentiment_roberta(feat_df["title"].tolist(), sentiment_pipeline)
    else:
        SENTIMENT_METHOD = "情感词典法（fallback）"
        log.warning("情感分析使用词典法（fallback）。")
        scores = [sentiment_dict_method(t) for t in feat_df["title"].tolist()]

    log.info("情感分析方法：" + SENTIMENT_METHOD)
    feat_df["title_sentiment_score"] = [round(s, 4) for s in scores]

    def score_to_valence(s):
        if s > 0.6:   return "positive"
        elif s < 0.4: return "negative"
        else:         return "neutral"

    feat_df["title_sentiment_valence"]   = feat_df["title_sentiment_score"].apply(score_to_valence)
    feat_df["title_sentiment_intensity"] = (feat_df["title_sentiment_score"] - 0.5).abs().round(4)

    # ── 5.5 clickbait_score ───────────────────────────────────────────────────
    log.info("计算 title_clickbait_score...")
    feat_df["title_clickbait_score"] = compute_clickbait_score(feat_df)

    # ── 5.6 排列列顺序 ────────────────────────────────────────────────────────
    FEATURE_COLS = [
        "bvid", "title",
        "title_char_len", "title_word_count", "title_char_len_sq", "title_punctuation_count",
        "title_question_mark", "title_exclamation_mark",
        "title_question_type", "title_exclamation_type",
        "title_has_number", "title_number_count",
        "title_has_quantifier", "title_number_phrase",
        "curiosity_marker_count", "title_has_curiosity", "title_clickbait_score",
        "practical_value_word_count", "title_has_practical_value",
        "title_sentiment_score", "title_sentiment_valence", "title_sentiment_intensity",
    ]
    feat_df = feat_df[FEATURE_COLS]

    # ── 5.7 质量检查日志 ──────────────────────────────────────────────────────
    log.info("── 质量检查报告 ──")
    log.info("总行数：" + str(len(feat_df)))
    log.info("title_char_len 均值：" + str(round(feat_df["title_char_len"].mean(),2)) +
             "，中位数：" + str(feat_df["title_char_len"].median()) +
             "，最大：" + str(feat_df["title_char_len"].max()))
    log.info("疑问式标题比例：" + str(round(feat_df["title_question_type"].mean()*100,1)) + "%")
    log.info("感叹式标题比例：" + str(round(feat_df["title_exclamation_type"].mean()*100,1)) + "%")
    log.info("含悬念词比例：" + str(round(feat_df["title_has_curiosity"].mean()*100,1)) + "%")
    log.info("含实用价值词比例：" + str(round(feat_df["title_has_practical_value"].mean()*100,1)) + "%")
    log.info("含数字比例：" + str(round(feat_df["title_has_number"].mean()*100,1)) + "%")
    log.info("情感得分均值：" + str(round(feat_df["title_sentiment_score"].mean(),4)))
    valence_dist = feat_df["title_sentiment_valence"].value_counts(normalize=True) * 100
    log.info("情感极性分布（%）：\n" + valence_dist.round(1).to_string())

    # ── 5.8 保存 text_features.csv ────────────────────────────────────────────
    out_csv = OUT_DIR / "text_features.csv"
    feat_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log.info("已保存：" + str(out_csv))

    # ── 5.9 变量说明表 ────────────────────────────────────────────────────────
    var_dict = [
        {"变量名": "bvid",   "中文别名": "视频唯一标识符",  "类型": "字符串",
         "计算方法": "原始字段", "取值范围": "BVxxxx格式", "备注": "主键"},
        {"变量名": "title",  "中文别名": "视频标题",        "类型": "字符串",
         "计算方法": "原始字段", "取值范围": "中文文本", "备注": ""},

        {"变量名": "title_char_len",   "中文别名": "标题字符长度",
         "类型": "连续", "计算方法": "len(title)",
         "取值范围": "正整数", "备注": "含标点和空格"},
        {"变量名": "title_word_count", "中文别名": "标题词数（分词后）",
         "类型": "连续", "计算方法": "jieba.lcut后去空词计数",
         "取值范围": "正整数", "备注": "使用jieba精确模式"},
        {"变量名": "title_char_len_sq","中文别名": "标题字符长度平方项",
         "类型": "连续", "计算方法": "title_char_len ** 2",
         "取值范围": "正整数", "备注": "用于检验非线性效应"},
        {"变量名": "title_punctuation_count","中文别名": "标题标点数量",
         "类型": "连续", "计算方法": "正则匹配中英文标点",
         "取值范围": "非负整数", "备注": ""},

        {"变量名": "title_question_mark","中文别名": "含问号",
         "类型": "二元(0/1)", "计算方法": "是否含?或？",
         "取值范围": "0/1", "备注": ""},
        {"变量名": "title_exclamation_mark","中文别名": "含感叹号",
         "类型": "二元(0/1)", "计算方法": "是否含!或！",
         "取值范围": "0/1", "备注": ""},
        {"变量名": "title_question_type","中文别名": "疑问式标题",
         "类型": "二元(0/1)", "计算方法": "含问号 OR 含疑问词（为什么/怎么/如何等）",
         "取值范围": "0/1", "备注": "疑问词列表见代码QUESTION_WORDS"},
        {"变量名": "title_exclamation_type","中文别名": "感叹式标题",
         "类型": "二元(0/1)", "计算方法": "等同title_exclamation_mark",
         "取值范围": "0/1", "备注": ""},

        {"变量名": "title_has_number","中文别名": "含数字",
         "类型": "二元(0/1)", "计算方法": "含阿拉伯数字或中文数字",
         "取值范围": "0/1", "备注": ""},
        {"变量名": "title_number_count","中文别名": "数字表达数量",
         "类型": "连续", "计算方法": "阿拉伯数字串数+中文数字串数",
         "取值范围": "非负整数", "备注": ""},
        {"变量名": "title_has_quantifier","中文别名": "含量词",
         "类型": "二元(0/1)", "计算方法": "是否含量词列表中的词",
         "取值范围": "0/1", "备注": "量词列表见代码QUANTIFIERS"},
        {"变量名": "title_number_phrase","中文别名": "数字短语列表",
         "类型": "字符串", "计算方法": "正则提取数字+量词短语",
         "取值范围": "短语字符串（|分隔）", "备注": "如 3分钟|10个"},

        {"变量名": "curiosity_marker_count","中文别名": "悬念词数量",
         "类型": "连续", "计算方法": "统计CURIOSITY_WORDS词典命中数",
         "取值范围": "非负整数", "备注": "词典见代码CURIOSITY_WORDS（37词）"},
        {"变量名": "title_has_curiosity","中文别名": "含悬念词",
         "类型": "二元(0/1)", "计算方法": "curiosity_marker_count>0",
         "取值范围": "0/1", "备注": ""},
        {"变量名": "title_clickbait_score","中文别名": "悬念表达强度（可解释合成指标）",
         "类型": "连续[0,1]",
         "计算方法": "0.40*悬念词归一化 + 0.25*疑问句 + 0.20*感叹句 + 0.15*情感强度，整体min-max归一化",
         "取值范围": "[0,1]", "备注": "值越大悬念/点击引导倾向越强"},

        {"变量名": "practical_value_word_count","中文别名": "实用价值词数量",
         "类型": "连续", "计算方法": "统计PRACTICAL_WORDS词典命中数",
         "取值范围": "非负整数", "备注": "词典见代码PRACTICAL_WORDS（35词）"},
        {"变量名": "title_has_practical_value","中文别名": "含实用价值词",
         "类型": "二元(0/1)", "计算方法": "practical_value_word_count>0",
         "取值范围": "0/1", "备注": ""},

        {"变量名": "title_sentiment_score","中文别名": "标题情感得分",
         "类型": "连续[0,1]",
         "计算方法": "优先RoBERTa模型；不可用时使用正负词典法（正负词计数）",
         "取值范围": "[0,1]，>0.5偏正向，<0.5偏负向", "备注": ""},
        {"变量名": "title_sentiment_valence","中文别名": "情感极性分类",
         "类型": "类别", "计算方法": ">0.6为positive，<0.4为negative，其余neutral",
         "取值范围": "positive/neutral/negative", "备注": ""},
        {"变量名": "title_sentiment_intensity","中文别名": "情感极性强度",
         "类型": "连续[0,0.5]", "计算方法": "abs(title_sentiment_score - 0.5)",
         "取值范围": "[0, 0.5]", "备注": "越高则情感越极端（无论正负）"},
    ]

    var_df = pd.DataFrame(var_dict)
    var_path = OUT_DIR / "variable_dictionary_step3.xlsx"
    with pd.ExcelWriter(var_path, engine="openpyxl") as writer:
        var_df.to_excel(writer, index=False, sheet_name="变量说明")
        ws = writer.sheets["变量说明"]
        for col_cells in ws.columns:
            max_len = max(len(str(c.value or "")) for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 60)
    log.info("已保存变量说明表：" + str(var_path))

    # ── 5.10 抽样质检表 ───────────────────────────────────────────────────────
    log.info("生成抽样质检表（100条）...")
    sample_n = min(100, len(feat_df))
    sample_df = feat_df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    sample_df["jieba_words"] = sample_df["title"].apply(
        lambda t: " | ".join(jieba.lcut(str(t)))
    )
    sample_df["curiosity_words_hit"] = sample_df["title"].apply(
        lambda t: ",".join([w for w in CURIOSITY_WORDS if w in str(t)])
    )
    sample_df["practical_words_hit"] = sample_df["title"].apply(
        lambda t: ",".join([w for w in PRACTICAL_WORDS if w in str(t)])
    )

    audit_cols = [
        "bvid", "title", "title_char_len", "title_word_count",
        "jieba_words",
        "title_question_type", "title_exclamation_type",
        "curiosity_marker_count", "title_has_curiosity", "curiosity_words_hit",
        "practical_value_word_count", "title_has_practical_value", "practical_words_hit",
        "title_sentiment_score", "title_sentiment_valence",
        "title_clickbait_score",
    ]
    audit_df = sample_df[audit_cols]

    audit_path = LOG_DIR / "title_feature_audit_samples.xlsx"
    with pd.ExcelWriter(audit_path, engine="openpyxl") as writer:
        audit_df.to_excel(writer, index=False, sheet_name="抽样质检")
        ws = writer.sheets["抽样质检"]
        for col_cells in ws.columns:
            max_len = max(len(str(c.value or "")) for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 50)
    log.info("已保存质检表：" + str(audit_path))

    # ── 5.11 可视化 ───────────────────────────────────────────────────────────
    def get_fp():
        if FONT_PATH:
            return fm.FontProperties(fname=FONT_PATH)
        return None
    fp = get_fp()

    # --- 图1：标题长度分布 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#F8F9FA")

    ax1 = axes[0]
    ax1.set_facecolor("#FFFFFF")
    ax1.hist(feat_df["title_char_len"], bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax1.axvline(feat_df["title_char_len"].mean(), color="#DD4444", linestyle="--", linewidth=1.5)
    ax1.axvline(feat_df["title_char_len"].median(), color="#44AA44", linestyle=":", linewidth=1.5)
    mean_val  = round(feat_df["title_char_len"].mean(), 1)
    med_val   = round(feat_df["title_char_len"].median(), 1)

    legend_labels = ["均值=" + str(mean_val), "中位数=" + str(med_val)]
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color="#DD4444", linestyle="--", linewidth=1.5),
        Line2D([0],[0], color="#44AA44", linestyle=":",  linewidth=1.5),
    ]
    if fp:
        ax1.set_xlabel("标题字符长度", fontproperties=fp, fontsize=12)
        ax1.set_ylabel("视频数量",     fontproperties=fp, fontsize=12)
        ax1.set_title("标题字符长度分布", fontproperties=fp, fontsize=14, fontweight="bold")
        ax1.legend(handles, legend_labels, prop=fp)
    else:
        ax1.set_xlabel("Title Character Length", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title("Title Length Distribution", fontsize=14, fontweight="bold")
        ax1.legend(handles, ["Mean="+str(mean_val), "Median="+str(med_val)])
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    if "domain" in df.columns:
        domain_map = df[["bvid", "domain"]].drop_duplicates()
        plot_df = feat_df.merge(domain_map, on="bvid", how="left")
        domains = sorted(plot_df["domain"].dropna().unique().tolist())
        word_data = [plot_df[plot_df["domain"]==d]["title_word_count"].dropna().values for d in domains]
        bp = ax2.boxplot(word_data, patch_artist=True, notch=False,
                         medianprops=dict(color="black", linewidth=2))
        colors_box = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        for patch, color in zip(bp["boxes"], colors_box[:len(domains)]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax2.set_xticks(range(1, len(domains)+1))
        if fp:
            ax2.set_xticklabels(domains, fontproperties=fp, fontsize=9, rotation=15)
            ax2.set_ylabel("词数（分词后）", fontproperties=fp, fontsize=12)
            ax2.set_title("各领域标题词数分布",  fontproperties=fp, fontsize=14, fontweight="bold")
        else:
            ax2.set_xticklabels(domains, fontsize=9, rotation=15)
            ax2.set_ylabel("Word Count", fontsize=12)
            ax2.set_title("Word Count by Domain", fontsize=14, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.hist(feat_df["title_word_count"], bins=30, color="#DD8452", edgecolor="white")

    plt.tight_layout(pad=2.0)
    fig1_path = FIG_DIR / "title_length_distribution.png"
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("已保存图表：" + str(fig1_path))

    # --- 图2：悬念词/实用词/疑问/感叹占比 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#F8F9FA")

    if "domain" in df.columns:
        domain_map = df[["bvid", "domain"]].drop_duplicates()
        plot_df2   = feat_df.merge(domain_map, on="bvid", how="left")
        domains2   = sorted(plot_df2["domain"].dropna().unique().tolist())

        curiosity_rates   = []
        practical_rates   = []
        question_rates    = []
        exclamation_rates = []
        for d in domains2:
            sub = plot_df2[plot_df2["domain"]==d]
            curiosity_rates.append(sub["title_has_curiosity"].mean()*100)
            practical_rates.append(sub["title_has_practical_value"].mean()*100)
            question_rates.append(sub["title_question_type"].mean()*100)
            exclamation_rates.append(sub["title_exclamation_type"].mean()*100)

        x     = np.arange(len(domains2))
        width = 0.35
        pal   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        ax3 = axes[0]
        ax3.set_facecolor("#FFFFFF")
        b1 = ax3.bar(x - width/2, curiosity_rates, width, label="含悬念词",    color=pal[0], alpha=0.85)
        b2 = ax3.bar(x + width/2, practical_rates, width, label="含实用价值词", color=pal[1], alpha=0.85)
        for bar in list(b1)+list(b2):
            h = bar.get_height()
            kw = {"fontproperties": fp} if fp else {}
            ax3.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=9, **kw)
        ax3.set_xticks(x)
        if fp:
            ax3.set_xticklabels(domains2, fontproperties=fp, fontsize=9, rotation=10)
            ax3.set_ylabel("占比（%）", fontproperties=fp, fontsize=12)
            ax3.set_title("各领域悬念词/实用词出现比例", fontproperties=fp, fontsize=13, fontweight="bold")
            ax3.legend(prop=fp)
        else:
            ax3.set_xticklabels(domains2, fontsize=9, rotation=10)
            ax3.set_ylabel("Rate (%)", fontsize=12)
            ax3.set_title("Curiosity/Practical Word Rate", fontsize=13, fontweight="bold")
            ax3.legend()
        ax3.set_ylim(0, max(max(curiosity_rates), max(practical_rates))+15)
        ax3.grid(axis="y", alpha=0.3)

        ax4 = axes[1]
        ax4.set_facecolor("#FFFFFF")
        b3 = ax4.bar(x - width/2, question_rates,    width, label="疑问式标题", color=pal[2], alpha=0.85)
        b4 = ax4.bar(x + width/2, exclamation_rates, width, label="感叹式标题", color=pal[3], alpha=0.85)
        for bar in list(b3)+list(b4):
            h = bar.get_height()
            kw = {"fontproperties": fp} if fp else {}
            ax4.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=9, **kw)
        ax4.set_xticks(x)
        if fp:
            ax4.set_xticklabels(domains2, fontproperties=fp, fontsize=9, rotation=10)
            ax4.set_ylabel("占比（%）", fontproperties=fp, fontsize=12)
            ax4.set_title("各领域疑问式/感叹式标题比例", fontproperties=fp, fontsize=13, fontweight="bold")
            ax4.legend(prop=fp)
        else:
            ax4.set_xticklabels(domains2, fontsize=9, rotation=10)
            ax4.set_ylabel("Rate (%)", fontsize=12)
            ax4.set_title("Question/Exclamation Rate by Domain", fontsize=13, fontweight="bold")
            ax4.legend()
        ax4.set_ylim(0, max(max(question_rates), max(exclamation_rates))+15)
        ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout(pad=2.0)
    fig2_path = FIG_DIR / "title_curiosity_practical_rate.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("已保存图表：" + str(fig2_path))

    # ── 5.12 最终汇总 ─────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("【最终统计摘要】")
    log.info("有效标题数：" + str(len(feat_df)))
    log.info("情感分析方法：" + SENTIMENT_METHOD)
    num_cols = ["title_char_len","title_word_count","title_punctuation_count",
                "curiosity_marker_count","practical_value_word_count",
                "title_sentiment_score","title_sentiment_intensity","title_clickbait_score"]
    desc = feat_df[num_cols].describe().round(3)
    log.info("\n" + desc.to_string())
    log.info("=" * 70)
    log.info("第三步运行完成。")
    log.info("输出文件汇总：")
    log.info("  1. " + str(out_csv))
    log.info("  2. " + str(var_path))
    log.info("  3. " + str(audit_path))
    log.info("  4. " + str(fig1_path))
    log.info("  5. " + str(fig2_path))
    log.info("  6. " + str(LOG_FILE))


if __name__ == "__main__":
    main()
