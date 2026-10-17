# -*- coding: utf-8 -*-
"""
批量下载 B站视频封面图片

输入：
video_full_data_2025gpt_gpt.csv

要求字段：
bvid, cover_url, domain, level, title, tags, description, up_name

输出：
covers/{领域}/{层级}/{bvid}.jpg

失败记录：
cover_download_gpt_failed.csv

功能：
1. 自动读取 CSV / TSV
2. 自动创建 covers/{领域}/{层级}/ 多级目录
3. 自动处理 //i0.hdslb.com/... 形式的封面 URL
4. 添加 B站防盗链请求头，避免 403
5. 已下载且大于 1KB，并通过 PIL 校验的图片自动跳过
6. 下载失败自动重试 5 次
7. 支持 .part 临时文件和 Range 断点续传
8. 下载后使用 PIL 完整性验证，损坏则删除并重试
9. 每处理 200 张输出一次统计
"""

import os
import re
import time
import random
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from PIL import Image


# ============================================================
# 1. 基础配置
# ============================================================

# 如果 CSV 和脚本在同一目录，保持这样即可
INPUT_CSV = Path("video_full_data_2025gpt_gpt.csv")

# 如果 CSV 不在当前目录，可以改成绝对路径，例如：
# INPUT_CSV = Path(r"C:\Users\dlnZ0\PycharmProjects\pythonProject\video_full_data_2025gpt_gpt.csv")

OUTPUT_ROOT = Path("covers")
FAILED_CSV = Path("cover_download_gpt_failed.csv")

# 并发数。建议 6~16，不要太大，避免 CDN 限流
MAX_WORKERS = 12

# 每张图片最大重试次数
RETRY_TIMES = 5

# 每处理多少条输出一次统计
REPORT_EVERY = 200

# 小于等于 1KB 的文件视为异常图片或错误页
MIN_VALID_SIZE = 1024

# 分块下载大小
CHUNK_SIZE = 1024 * 64

# 请求超时：连接超时、读取超时
TIMEOUT = (8, 30)

# 是否按 bvid 去重
DROP_DUPLICATE_BVID = True


# ============================================================
# 2. B站图片请求头：防 403 关键
# ============================================================

HEADERS = {
    "Referer": "https://www.bilibili.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}


# ============================================================
# 3. 每个线程独立 Session
# ============================================================

_thread_local = threading.local()


def get_session() -> requests.Session:
    """
    每个线程使用独立 requests.Session，提高连接复用效率。
    """
    if not hasattr(_thread_local, "session"):
        session = requests.Session()

        adapter = HTTPAdapter(
            pool_connections=MAX_WORKERS * 2,
            pool_maxsize=MAX_WORKERS * 2,
            max_retries=0,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(HEADERS)

        _thread_local.session = session

    return _thread_local.session


# ============================================================
# 4. CSV 智能读取
# ============================================================

def read_csv_smart(path: Path) -> pd.DataFrame:
    """
    自动尝试常见编码和分隔符读取 CSV / TSV。

    重要修正：
    sep=None 时需要 engine='python'，
    但 engine='python' 不支持 low_memory=False，
    因此这里专门去掉 low_memory 参数。
    """

    if not path.exists():
        raise FileNotFoundError(
            f"找不到输入文件：{path.resolve()}\n"
            f"当前工作目录是：{Path.cwd()}\n"
            f"请确认 video_full_data_2025gpt_gpt.csv 是否放在脚本运行目录下。"
        )

    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    seps = [",", "\t", None]

    last_error = None
    last_columns = None

    for enc in encodings:
        for sep in seps:
            try:
                print(f"正在尝试读取：encoding={enc}, sep={repr(sep)}")

                if sep is None:
                    # 注意：engine='python' 不支持 low_memory=False
                    df = pd.read_csv(
                        path,
                        encoding=enc,
                        sep=None,
                        engine="python",
                    )
                else:
                    df = pd.read_csv(
                        path,
                        encoding=enc,
                        sep=sep,
                        low_memory=False,
                    )

                # 清理列名
                df.columns = [
                    str(c).strip().replace("\ufeff", "")
                    for c in df.columns
                ]

                last_columns = list(df.columns)
                print("识别到的字段：", last_columns[:30])

                if "bvid" in df.columns and "cover_url" in df.columns:
                    print(f"成功读取 CSV：encoding={enc}, sep={repr(sep)}")
                    return df
                else:
                    print("本次读取没有同时识别到 bvid 和 cover_url，继续尝试其他格式。")

            except Exception as e:
                last_error = e
                print(f"读取失败：encoding={enc}, sep={repr(sep)}，错误：{e}")

    raise RuntimeError(
        "CSV 读取失败，或未能识别 bvid / cover_url 字段。\n"
        f"最后一次识别到的字段：{last_columns}\n"
        f"最后错误：{last_error}"
    )


# ============================================================
# 5. 文本与路径工具
# ============================================================

def clean_text(x) -> str:
    """
    清洗文本。
    """
    if pd.isna(x):
        return ""
    return str(x).strip()


def sanitize_path_part(s: str, max_len: int = 80) -> str:
    """
    清洗 Windows / macOS / Linux 文件夹非法字符。
    """
    s = clean_text(s)

    if not s:
        s = "unknown"

    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("._ ")

    if not s:
        s = "unknown"

    return s[:max_len]


def normalize_cover_url(cover_url: str) -> str:
    """
    统一 B站封面 URL。

    支持：
    http://i0.hdslb.com/...
    https://i0.hdslb.com/...
    //i0.hdslb.com/...
    i0.hdslb.com/...
    """
    url = clean_text(cover_url)

    if not url:
        return ""

    if url.startswith("http://") or url.startswith("https://"):
        return url

    if url.startswith("//"):
        return "https:" + url

    if url.startswith("i") and "hdslb.com" in url:
        return "https://" + url

    return "https:" + url


def normalize_bvid(bvid: str) -> str:
    """
    清洗 bvid，作为文件名。
    """
    bvid = clean_text(bvid)
    bvid = re.sub(r"[^\w]", "", bvid)
    return bvid


# ============================================================
# 6. 领域细分逻辑
# ============================================================

FIELD_KEYWORDS = {
    "科学科普": [
        "科学", "科普", "医学", "神经科学", "大脑", "物理", "化学", "生物",
        "天文", "数学", "AI", "人工智能", "算法", "芯片", "科技", "数码",
        "摄影", "相机", "无人机", "机器人", "3D打印", "VR", "MR", "画质",
        "传感器", "高速摄影", "环境", "土壤", "生态", "工程", "技术",
        "评测", "器材", "手机", "iPhone", "大疆", "索尼", "佳能", "尼康",
        "哈苏", "适马", "松下", "vivo", "小米", "苹果", "影像", "电影机"
    ],

    "社科人文": [
        "社科", "人文", "历史", "文化", "哲学", "社会", "法律", "心理",
        "人格", "焦虑", "抑郁", "潜意识", "灵性", "身心灵", "疗愈",
        "显化", "传讯", "塔罗", "美国", "特朗普", "川普", "拜登",
        "俄罗斯", "乌克兰", "俄乌", "欧盟", "德国", "日本", "战争",
        "政治", "国际", "阶级", "种族", "城市", "社会观察局",
        "社会热点", "人生智慧", "心理健康", "信仰", "人性",
        "人文历史", "心理学", "哲学", "国家安全", "移民", "左翼"
    ],

    "财经商业": [
        "财经", "商业", "经济", "金融", "投资", "股票", "证券", "基金",
        "房价", "房地产", "关税", "贸易", "市场", "预算", "税",
        "电商", "创业", "品牌", "运营", "营收", "公司", "企业",
        "产业", "供应链", "通胀", "物价", "购买力", "赚钱", "成本",
        "利润", "管理", "商业模式"
    ],

    "职场教育": [
        "职场", "职业", "工作", "就业", "求职", "面试", "简历", "办公",
        "效率", "教程", "学习", "考试", "考研", "高考", "大学", "毕业",
        "校园", "论文", "代写", "技能", "软件", "剪辑", "口播",
        "自媒体", "创作", "工作流", "团队", "员工", "培训", "课程",
        "老师", "学生", "经验分享", "方法", "技巧", "入门", "新手"
    ],
}


def classify_field(row) -> str:
    """
    得到输出路径中的 {领域}。

    优先级：
    1. 如果存在前序任务生成的 knowledge_field_4，则优先使用；
    2. 如果 domain 已经是目标四类，则直接使用；
    3. 否则根据 title / tags / description / up_name 做关键词分类；
    4. 仍无法判断则使用 domain 或归入“其他”。
    """

    # 1. 如果已经有前序任务的分类列，优先使用
    for col in ["knowledge_field_4", "field_4", "领域细分", "细分领域"]:
        if col in row.index:
            value = clean_text(row.get(col))
            if value and value not in ["未匹配/非知识区", "未匹配", "None", "nan"]:
                return sanitize_path_part(value)

    domain = clean_text(row.get("domain", ""))

    # 2. domain 直接映射
    direct_map = {
        "科学科普": "科学科普",
        "社科人文": "社科人文",
        "人文历史": "社科人文",
        "社科·法律·心理": "社科人文",
        "财经商业": "财经商业",
        "职场教育": "职场教育",
        "职业职场": "职场教育",
        "校园学习": "职场教育",
    }

    if domain in direct_map:
        return direct_map[domain]

    # 3. 关键词分类
    title = clean_text(row.get("title", ""))
    tags = clean_text(row.get("tags", ""))
    desc = clean_text(row.get("description", ""))
    up_name = clean_text(row.get("up_name", ""))

    text = " ".join([
        title,
        title,
        title,
        tags,
        tags,
        tags,
        tags,
        desc,
        up_name,
        domain,
    ]).lower()

    scores = {field: 0 for field in FIELD_KEYWORDS.keys()}

    for field, keywords in FIELD_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text:
                scores[field] += 1

    best_field = max(scores, key=scores.get)
    best_score = scores[best_field]

    if best_score > 0:
        return best_field

    # 4. 兜底
    if domain:
        return sanitize_path_part(domain)

    return "其他"


def classify_level(row) -> str:
    """
    得到输出路径中的 {层级}。

    你的数据中 level 有 -1 / 0 / 1 等值。
    输出为：
    level_-1
    level_0
    level_1
    """
    raw_level = clean_text(row.get("level", ""))

    if raw_level == "":
        return "level_unknown"

    try:
        num = float(raw_level)
        if num.is_integer():
            raw_level = str(int(num))
    except Exception:
        pass

    return sanitize_path_part(f"level_{raw_level}")


def build_save_path(row) -> Path:
    """
    生成保存路径：
    covers/{领域}/{层级}/{bvid}.jpg
    """
    field = classify_field(row)
    level = classify_level(row)
    bvid = normalize_bvid(row.get("bvid", ""))

    if not bvid:
        bvid = "unknown_bvid"

    return OUTPUT_ROOT / field / level / f"{bvid}.jpg"


# ============================================================
# 7. 图片完整性验证
# ============================================================

def is_valid_image(path: Path) -> bool:
    """
    判断图片是否存在、大小是否 > 1KB，并通过 PIL 完整性验证。
    """
    try:
        if not path.exists():
            return False

        if path.stat().st_size <= MIN_VALID_SIZE:
            return False

        with Image.open(path) as img:
            img.verify()

        return True

    except Exception:
        return False


def remove_file_safely(path: Path):
    """
    安全删除文件。
    """
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


# ============================================================
# 8. 单次下载，支持 .part 和 Range 断点续传
# ============================================================

def download_once(url: str, dest_path: Path) -> int:
    """
    单次下载。

    使用 .part 临时文件。
    如果 .part 存在，则通过 Range 请求断点续传。
    下载完成后：
    1. 检查文件大小
    2. PIL verify 校验
    3. os.replace 原子替换为正式 jpg
    4. 再次校验正式文件

    返回下载后文件大小。
    """

    session = get_session()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    mode = "wb"
    request_headers = dict(HEADERS)

    # 如果存在 part 文件，则尝试断点续传
    if part_path.exists():
        local_size = part_path.stat().st_size
        if local_size > 0:
            request_headers["Range"] = f"bytes={local_size}-"
            mode = "ab"

    with session.get(
        url,
        headers=request_headers,
        stream=True,
        timeout=TIMEOUT,
        allow_redirects=True,
    ) as resp:

        status = resp.status_code

        if status == 206:
            # 服务器支持 Range
            mode = "ab"

        elif status == 200:
            # 服务器不支持 Range 或重新完整下载
            mode = "wb"

        elif status == 416:
            # Range 不合法，删除 part 后下次重来
            remove_file_safely(part_path)
            raise RuntimeError("HTTP 416 Range Not Satisfiable，已删除异常 part 文件")

        else:
            raise RuntimeError(f"HTTP {status}")

        content_type = resp.headers.get("Content-Type", "").lower()

        if content_type and "image" not in content_type:
            raise RuntimeError(f"响应不是图片，Content-Type={content_type}")

        with open(part_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    # 文件大小检查
    if not part_path.exists() or part_path.stat().st_size <= MIN_VALID_SIZE:
        remove_file_safely(part_path)
        raise RuntimeError("下载文件小于等于 1KB，疑似损坏、403错误页或空响应")

    # PIL 校验临时文件
    try:
        with Image.open(part_path) as img:
            img.verify()
    except Exception as e:
        remove_file_safely(part_path)
        raise RuntimeError(f"PIL 校验失败，已删除损坏图片：{e}")

    # 原子替换为正式文件
    os.replace(part_path, dest_path)

    # 二次校验正式文件
    if not is_valid_image(dest_path):
        remove_file_safely(dest_path)
        raise RuntimeError("正式文件二次校验失败，已删除")

    return dest_path.stat().st_size


# ============================================================
# 9. 带重试的下载任务
# ============================================================

def download_cover_task(row_dict: dict) -> dict:
    """
    单条视频封面下载任务。
    """
    bvid = normalize_bvid(row_dict.get("bvid", ""))
    raw_url = clean_text(row_dict.get("cover_url", ""))
    url = normalize_cover_url(raw_url)

    row = pd.Series(row_dict)

    field = classify_field(row)
    level = classify_level(row)
    dest_path = build_save_path(row)

    result = {
        "bvid": bvid,
        "cover_url": raw_url,
        "normalized_url": url,
        "field": field,
        "level": level,
        "save_path": str(dest_path),
        "title": clean_text(row_dict.get("title", "")),
        "status": "",
        "error": "",
        "bytes": 0,
    }

    if not bvid:
        result["status"] = "failed"
        result["error"] = "bvid 为空"
        return result

    if not url:
        result["status"] = "failed"
        result["error"] = "cover_url 为空"
        return result

    # 已存在且有效，直接跳过
    if dest_path.exists():
        if is_valid_image(dest_path):
            result["status"] = "skipped"
            result["bytes"] = dest_path.stat().st_size
            return result
        else:
            remove_file_safely(dest_path)

    last_error = ""

    for attempt in range(1, RETRY_TIMES + 1):
        try:
            file_size = download_once(url, dest_path)

            result["status"] = "downloaded"
            result["bytes"] = file_size
            return result

        except Exception as e:
            last_error = str(e)

            if attempt < RETRY_TIMES:
                # 间隔递增：2 / 4 / 6 / 8 秒
                sleep_seconds = 2 * attempt

                # 随机抖动，降低并发请求被限流概率
                sleep_seconds += random.uniform(0.1, 0.8)

                time.sleep(sleep_seconds)

    result["status"] = "failed"
    result["error"] = last_error
    return result


# ============================================================
# 10. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("B站视频封面批量下载脚本启动")
    print(f"当前工作目录：{Path.cwd()}")
    print(f"输入文件：{INPUT_CSV.resolve()}")
    print("=" * 60)

    df = read_csv_smart(INPUT_CSV)

    required_cols = ["bvid", "cover_url"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"输入 CSV 缺少必要字段：{col}")

    # 清理 bvid 和 cover_url
    df["bvid"] = df["bvid"].map(clean_text)
    df["cover_url"] = df["cover_url"].map(clean_text)

    before_count = len(df)

    # 去掉空 bvid 或空 cover_url
    df = df[
        (df["bvid"] != "") &
        (df["cover_url"] != "")
    ].copy()

    after_count = len(df)

    print(f"原始记录数：{before_count}")
    print(f"有效记录数：{after_count}")

    # 按 bvid 去重
    if DROP_DUPLICATE_BVID:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["bvid"], keep="first").copy()
        print(f"按 bvid 去重后：{len(df)}，去重 {before_dedup - len(df)} 条")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    records = df.to_dict(orient="records")

    stats = {
        "processed": 0,
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "bytes": 0,
    }

    failed_rows = []

    print("=" * 60)
    print(f"开始下载封面")
    print(f"并发数：{MAX_WORKERS}")
    print(f"输出目录：{OUTPUT_ROOT.resolve()}")
    print("=" * 60)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_cover_task, row)
            for row in records
        ]

        for future in as_completed(futures):
            try:
                res = future.result()
            except Exception as e:
                res = {
                    "bvid": "",
                    "cover_url": "",
                    "normalized_url": "",
                    "field": "",
                    "level": "",
                    "save_path": "",
                    "title": "",
                    "status": "failed",
                    "error": f"线程任务异常：{e}",
                    "bytes": 0,
                }

            stats["processed"] += 1
            stats["bytes"] += int(res.get("bytes", 0) or 0)

            status = res.get("status")

            if status == "downloaded":
                stats["downloaded"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
                failed_rows.append(res)

            if stats["processed"] % REPORT_EVERY == 0:
                elapsed = time.time() - start_time
                speed = stats["processed"] / elapsed if elapsed > 0 else 0

                print(
                    f"[统计] 已处理 {stats['processed']}/{len(records)} | "
                    f"新下载 {stats['downloaded']} | "
                    f"跳过 {stats['skipped']} | "
                    f"失败 {stats['failed']} | "
                    f"速度 {speed:.2f} 条/秒"
                )

    elapsed = time.time() - start_time

    # 写失败 CSV
    failed_columns = [
        "bvid",
        "cover_url",
        "normalized_url",
        "field",
        "level",
        "save_path",
        "title",
        "status",
        "error",
        "bytes",
    ]

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_df.to_csv(FAILED_CSV, index=False, encoding="utf-8-sig")
        print(f"失败记录已保存：{FAILED_CSV.resolve()}")
    else:
        failed_df = pd.DataFrame(columns=failed_columns)
        failed_df.to_csv(FAILED_CSV, index=False, encoding="utf-8-sig")
        print("无失败记录。已生成空的失败记录文件。")

    print("=" * 60)
    print("下载完成")
    print(f"总处理：{stats['processed']}")
    print(f"新下载：{stats['downloaded']}")
    print(f"已存在跳过：{stats['skipped']}")
    print(f"失败：{stats['failed']}")
    print(f"总耗时：{elapsed:.2f} 秒")

    if elapsed > 0:
        print(f"平均速度：{stats['processed'] / elapsed:.2f} 条/秒")

    print(f"输出目录：{OUTPUT_ROOT.resolve()}")
    print(f"失败记录：{FAILED_CSV.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()