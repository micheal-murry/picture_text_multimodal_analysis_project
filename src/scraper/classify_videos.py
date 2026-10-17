# -*- coding: utf-8 -*-
"""
功能：
对 B 站知识类视频进行四分区分类：
1. 科学科普
2. 社科人文
3. 财经商业
4. 职场教育

输入：
video_full_data_2025gpt_gpt.csv

输出：
video_full_data_2025gpt_gpt_classified.csv
分类结果写入最后一列：knowledge_field_4
"""

import re
import json
import time
from pathlib import Path

import pandas as pd
import requests


# =========================
# 1. 文件路径配置
# =========================

INPUT_CSV = Path("video_full_data_2025gpt_gpt.csv")
OUTPUT_CSV = Path("video_full_data_2025gpt_gpt_classified.csv")

# B站接口缓存文件，避免重复请求
CACHE_FILE = Path("bili_tid_cache.json")

# 是否通过 bvid 请求 B 站接口获取官方 tid/tname
FETCH_TID_FROM_BILI = True

# 请求间隔，避免过快访问
REQUEST_SLEEP = 0.25

# 最后一列列名
OUT_COL = "knowledge_field_4"

# 如果无法判断，输出什么
UNMATCHED_LABEL = "未匹配/非知识区"


# =========================
# 2. B站知识区 tid 映射
# =========================
"""
B站分区 tid 在不同资料或历史版本中可能略有差异。

较常见的知识区子分区包括：
- 科学科普
- 社科·法律·心理
- 人文历史
- 财经商业
- 校园学习
- 职业职场
- 设计·创意
- 野生技能协会

本脚本优先使用 tname 精准匹配；
如果只有 tid，则使用 TID_TO_FIELD_CURRENT 映射。
"""

# 按 B 站 tname 匹配，优先级最高
TNAME_TO_FIELD = {
    "科学科普": "科学科普",
    "医学科普": "科学科普",

    "人文历史": "社科人文",
    "社科·法律·心理": "社科人文",
    "社科法律心理": "社科人文",
    "社会科学": "社科人文",
    "法律": "社科人文",
    "心理": "社科人文",

    "财经商业": "财经商业",

    "职业职场": "职场教育",
    "校园学习": "职场教育",
    "野生技能协会": "职场教育",
    "野生技术协会": "职场教育",
    "设计·创意": "职场教育",
    "设计创意": "职场教育",
}

# 常见 B站知识区 tid 映射
TID_TO_FIELD_CURRENT = {
    201: "科学科普",

    124: "社科人文",   # 社科·法律·心理
    228: "社科人文",   # 人文历史
    236: "社科人文",   # 兼容部分资料中的社科法律心理

    207: "财经商业",   # 财经商业

    208: "职场教育",   # 校园学习
    209: "职场教育",   # 职业职场
    122: "职场教育",   # 野生技能协会
    229: "职场教育",   # 设计·创意
}

# 如果你的爬虫或论文口径明确采用题目中给出的 tid 参考表，
# 可把 USE_REFERENCE_TID_TABLE 改为 True
USE_REFERENCE_TID_TABLE = False

TID_TO_FIELD_REFERENCE = {
    124: "科学科普",
    122: "社科人文",
    236: "社科人文",
    207: "财经商业",
    208: "职场教育",
}

TID_TO_FIELD = (
    TID_TO_FIELD_REFERENCE
    if USE_REFERENCE_TID_TABLE
    else TID_TO_FIELD_CURRENT
)


# =========================
# 3. 关键词兜底分类词典
# =========================
"""
当数据中没有 tid，或者 B站接口访问失败时，
使用 title/tags/description/up_name/domain 进行快速分类。
"""

KEYWORDS = {
    "科学科普": {
        "strong": [
            "科学科普", "科普", "科学", "医学", "疾病", "临床", "神经科学",
            "大脑", "物理", "化学", "生物", "天文", "数学", "实验",
            "AI", "人工智能", "算法", "芯片", "科技", "数码", "技术",
            "工程", "无人机", "机器人", "相机", "摄影", "影像", "画质",
            "光学", "高速摄影", "传感器", "3D打印", "VR", "MR", "环境",
            "生态", "土壤", "地球", "航天"
        ],
        "weak": [
            "健康", "医院", "药", "研究", "数据", "评测", "电子", "屏幕",
            "手机", "汽车", "电池", "航拍", "器材", "样片"
        ],
    },

    "社科人文": {
        "strong": [
            "社科", "社会", "法律", "心理", "哲学", "人文", "历史",
            "文化", "地理", "考古", "博物馆", "政治", "国际", "美国",
            "特朗普", "川普", "拜登", "俄罗斯", "乌克兰", "俄乌",
            "战争", "冲突", "德国", "日本", "欧盟", "阶级", "种族",
            "宗教", "城市", "社会观察局", "人格", "焦虑", "抑郁",
            "潜意识", "灵性", "身心灵", "疗愈", "显化", "传讯", "塔罗"
        ],
        "weak": [
            "人生", "情感", "家庭", "教育", "信仰", "伦理", "社会热点",
            "心理健康", "人性", "民权", "民主", "政党", "国家安全"
        ],
    },

    "财经商业": {
        "strong": [
            "财经", "商业", "经济", "金融", "投资", "股票", "证券", "基金",
            "房价", "房地产", "关税", "贸易", "市场", "预算", "税",
            "营收", "电商", "创业", "品牌", "运营", "产业", "供应链",
            "通胀", "物价", "购买力", "公司", "企业", "管理"
        ],
        "weak": [
            "赚钱", "消费", "成本", "利润", "价格", "资本", "并购", "商业模式"
        ],
    },

    "职场教育": {
        "strong": [
            "职场", "职业", "工作", "就业", "求职", "面试", "简历",
            "办公", "效率", "教程", "学习", "考试", "考研", "高考",
            "大学", "毕业", "校园", "论文", "代写", "技能", "软件",
            "剪辑", "口播", "自媒体", "创作技巧", "员工", "团队",
            "工作流", "培训", "课程", "老师", "学生"
        ],
        "weak": [
            "经验分享", "方法", "技巧", "工具", "如何", "入门", "新手",
            "职场教育", "职业规划"
        ],
    },
}


# =========================
# 4. 工具函数
# =========================

def read_csv_smart(path: Path) -> pd.DataFrame:
    """
    自动尝试常见编码读取 CSV。
    """
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"成功读取文件：{path}，编码：{enc}")
            return df
        except Exception as e:
            last_error = e

    raise RuntimeError(f"CSV 读取失败，请检查文件编码或格式。最后错误：{last_error}")


def clean_text(x) -> str:
    """
    清洗文本字段。
    """
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_cache() -> dict:
    """
    读取 B站 tid 缓存。
    """
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: dict):
    """
    保存 B站 tid 缓存。
    """
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def fetch_bili_tid_tname(bvid: str, session: requests.Session, cache: dict):
    """
    通过 B站公开视频接口，根据 bvid 获取 tid 和 tname。
    接口：
    https://api.bilibili.com/x/web-interface/view?bvid=BVxxxx
    """
    bvid = clean_text(bvid)

    if not bvid:
        return None, None

    if bvid in cache:
        item = cache[bvid]
        return item.get("tid"), item.get("tname")

    url = "https://api.bilibili.com/x/web-interface/view"

    try:
        resp = session.get(url, params={"bvid": bvid}, timeout=10)
        data = resp.json()

        if data.get("code") == 0 and isinstance(data.get("data"), dict):
            video_data = data["data"]
            tid = video_data.get("tid")
            tname = video_data.get("tname")

            cache[bvid] = {
                "tid": tid,
                "tname": tname,
            }
        else:
            tid, tname = None, None
            cache[bvid] = {
                "tid": None,
                "tname": None,
                "error": data.get("message"),
            }

    except Exception as e:
        tid, tname = None, None
        cache[bvid] = {
            "tid": None,
            "tname": None,
            "error": str(e),
        }

    time.sleep(REQUEST_SLEEP)
    return tid, tname


def classify_by_tname(tname: str):
    """
    根据 B站官方 tname 精准分类。
    """
    tname = clean_text(tname)
    if not tname:
        return None

    # 完全匹配优先
    if tname in TNAME_TO_FIELD:
        return TNAME_TO_FIELD[tname]

    # 包含匹配
    for key, field in TNAME_TO_FIELD.items():
        if key in tname:
            return field

    return None


def classify_by_tid(tid):
    """
    根据 B站官方 tid 分类。
    """
    if pd.isna(tid):
        return None

    try:
        tid_int = int(float(tid))
    except Exception:
        return None

    return TID_TO_FIELD.get(tid_int)


def keyword_score(text: str) -> dict:
    """
    对拼接后的文本进行关键词打分。
    """
    text = clean_text(text)
    text_lower = text.lower()

    scores = {
        "科学科普": 0,
        "社科人文": 0,
        "财经商业": 0,
        "职场教育": 0,
    }

    for field, kw_group in KEYWORDS.items():
        for kw in kw_group["strong"]:
            if kw.lower() in text_lower:
                scores[field] += 3

        for kw in kw_group["weak"]:
            if kw.lower() in text_lower:
                scores[field] += 1

    return scores


def classify_by_text(row) -> str:
    """
    根据已有字段进行关键词兜底分类。
    """
    domain = clean_text(row.get("domain", ""))
    title = clean_text(row.get("title", ""))
    tags = clean_text(row.get("tags", ""))
    desc = clean_text(row.get("description", ""))
    up_name = clean_text(row.get("up_name", ""))

    # 数据原始 domain 如果已经是目标四类之一，给予高权重
    domain_direct_map = {
        "科学科普": "科学科普",
        "社科人文": "社科人文",
        "人文历史": "社科人文",
        "社科·法律·心理": "社科人文",
        "财经商业": "财经商业",
        "职场教育": "职场教育",
        "职业职场": "职场教育",
        "校园学习": "职场教育",
    }

    # 拼接文本：
    # title 和 tags 更能代表主题，所以重复加权
    text = " ".join([
        title * 3,
        tags * 4,
        domain * 2,
        up_name,
        desc,
    ])

    scores = keyword_score(text)

    if domain in domain_direct_map:
        scores[domain_direct_map[domain]] += 5

    # 如果标题或标签出现非常明确的心理/历史/国际政治类词汇，偏向社科人文
    social_special = [
        "心理", "人格", "焦虑", "抑郁", "历史", "哲学", "社会",
        "国际", "美国", "特朗普", "川普", "俄乌", "乌克兰",
        "灵性", "身心灵", "疗愈", "潜意识"
    ]
    if any(w in text for w in social_special):
        scores["社科人文"] += 2

    # 取最高分
    best_field = max(scores, key=scores.get)
    best_score = scores[best_field]

    # 分数太低则认为无法判断
    if best_score < 2:
        return UNMATCHED_LABEL

    return best_field


def classify_row(row, api_info: dict):
    """
    单条视频分类逻辑：
    1. 优先使用数据中已有 tname
    2. 再使用接口获取到的 tname
    3. 再使用数据中已有 tid
    4. 再使用接口获取到的 tid
    5. 最后使用文本关键词兜底
    """

    # 1. 原始数据中如果已经有 tname/typename
    raw_tname = None
    for col in ["tname", "typename", "type_name", "分区名"]:
        if col in row.index:
            raw_tname = row.get(col)
            break

    field = classify_by_tname(raw_tname)
    if field:
        return field

    # 2. 接口获取到的 tname
    bvid = clean_text(row.get("bvid", ""))
    if bvid in api_info:
        api_tname = api_info[bvid].get("tname")
        field = classify_by_tname(api_tname)
        if field:
            return field

    # 3. 原始数据中如果已经有 tid/typeid
    raw_tid = None
    for col in ["tid", "typeid", "type_id", "分区id"]:
        if col in row.index:
            raw_tid = row.get(col)
            break

    field = classify_by_tid(raw_tid)
    if field:
        return field

    # 4. 接口获取到的 tid
    if bvid in api_info:
        api_tid = api_info[bvid].get("tid")
        field = classify_by_tid(api_tid)
        if field:
            return field

    # 5. 关键词兜底
    return classify_by_text(row)


# =========================
# 5. 主流程
# =========================

def main():
    df = read_csv_smart(INPUT_CSV)

    print("数据字段：")
    print(list(df.columns))

    # 去除列名前后的空格
    df.columns = [str(c).strip() for c in df.columns]

    api_info = {}

    # 如果需要，通过 bvid 获取官方 tid/tname
    if FETCH_TID_FROM_BILI and "bvid" in df.columns:
        cache = load_cache()

        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com/",
        })

        bvids = (
            df["bvid"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        print(f"准备通过 B站接口获取 tid/tname，共 {len(bvids)} 个唯一 bvid。")

        for i, bvid in enumerate(bvids, start=1):
            tid, tname = fetch_bili_tid_tname(bvid, session, cache)

            api_info[bvid] = {
                "tid": tid,
                "tname": tname,
            }

            if i % 50 == 0:
                save_cache(cache)
                print(f"已处理 {i}/{len(bvids)} 个 bvid")

        save_cache(cache)
        print("B站 tid/tname 获取完成。")

    else:
        print("未启用 B站接口获取，或数据中没有 bvid 字段，将使用本地字段进行分类。")

    # 如果已经存在输出列，先删除，保证最后一列是新结果
    if OUT_COL in df.columns:
        df = df.drop(columns=[OUT_COL])

    # 分类
    print("开始分类...")
    df[OUT_COL] = df.apply(lambda row: classify_row(row, api_info), axis=1)

    # 确保分类列位于最后一列
    cols = [c for c in df.columns if c != OUT_COL] + [OUT_COL]
    df = df[cols]

    # 输出
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"分类完成，已输出到：{OUTPUT_CSV}")
    print("分类结果分布：")
    print(df[OUT_COL].value_counts(dropna=False))


if __name__ == "__main__":
    main()