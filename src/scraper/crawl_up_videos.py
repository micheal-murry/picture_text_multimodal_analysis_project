#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B站UP主2025年视频全量数据爬取脚本
======================================
特性：
  - WBI 签名（应对B站2024+新版反爬）
  - 412风控指数退避重试 + 接口降级
  - 断点续传（跳过已爬bvid / 已完成UID）
  - 单IP并发≤3 / 随机请求间隔 / UP间暂停
  - 每爬完5个UP主自动备份CSV
  - 完整错误日志

依赖安装：
  pip install aiohttp pandas openpyxl
"""

import asyncio
import aiohttp
import pandas as pd
import csv
import json
import logging
import os
import random
import re
import time
import hashlib
import shutil
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Optional

# ============================================================
# 配置区
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR   = Path(os.environ.get("BILI_DATA_DIR", str(REPO_ROOT / "data" / "raw")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILE    = str(DATA_DIR / "up_list_final.xlsx")
OUTPUT_FILE   = str(DATA_DIR / "video_full_data_2025.csv")
ERROR_LOG     = str(DATA_DIR / "error_log.txt")
PROGRESS_FILE = str(DATA_DIR / "crawl_progress.json")

# Cookie 仅从环境变量读取，严禁把账号凭据写进代码（PowerShell: \="..."）
COOKIE = os.environ.get("BILI_COOKIE", "")

# 2025 时间戳范围
START_TS = int(datetime(2025, 1, 1,  0,  0,  0).timestamp())
END_TS   = int(datetime(2025, 12, 31, 23, 59, 59).timestamp())

MAX_CONCURRENT  = 3          # 最大并发数
REQ_INTERVAL    = (1.5, 3.5) # 请求间随机延迟(秒)
UP_INTERVAL     = 5          # 每个UP完成后暂停(秒)
BACKUP_EVERY    = 5          # 每N个UP备份一次
MAX_RETRY       = 6          # 最大重试次数
RETRY_BASE_SEC  = 15         # 412 退避基数(秒)
PAGE_SIZE       = 50         # 每页视频数(B站最大50)

# 精准领域映射（B站官方分区tid）
TID_DOMAIN_MAP = {
    124: "科学科普",
    123: "社科人文",
    207: "财经商业",
    208: "职场教育",
}

# UP主层级划分
def classify_level(fans: int) -> str:
    if fans >= 1_000_000:
        return "头部"
    elif fans >= 100_000:
        return "腰部"
    else:
        return "尾部"

# 合集正则
COLLECTION_RE = re.compile(
    r'第\s*[一二三四五六七八九十百千\d]+\s*[集期讲节话]'
    r'|合集|系列|[Ee][Pp]\s*\d+|全集|连载|第\s*\d+\s*[Pp]',
    re.IGNORECASE
)

# User-Agent 池
UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

# ============================================================
# 日志设置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(ERROR_LOG, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("bilibili_crawler")

# ============================================================
# WBI 签名实现（B站 2023+ 反爬必须）
# ============================================================
_MIXIN_KEY_ENC_TAB = [
    46, 47, 18,  2, 53,  8, 23, 32, 15, 50, 10, 31, 58,  3, 45, 35,
    27, 43,  5, 49, 33,  9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48,  7, 16, 24, 55, 40, 61, 26, 17,  0,  1, 60, 51, 30,  4,
    22, 25, 54, 21, 56, 59,  6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]

def _get_mixin_key(orig: str) -> str:
    return reduce(lambda s, i: s + orig[i], _MIXIN_KEY_ENC_TAB, "")[:32]

def wbi_sign(params: dict, img_key: str, sub_key: str) -> dict:
    """对参数进行WBI签名，返回带w_rid和wts的新参数字典"""
    mixin_key = _get_mixin_key(img_key + sub_key)
    params = dict(params)
    params["wts"] = round(time.time())
    params = dict(sorted(params.items()))
    # 过滤特殊字符
    _FILTER = set("!'()*")
    query = "&".join(
        f'{k}={"".join(c for c in str(v) if c not in _FILTER)}'
        for k, v in params.items()
    )
    params["w_rid"] = hashlib.md5((query + mixin_key).encode()).hexdigest()
    return params


# ============================================================
# 主爬虫类
# ============================================================
class BilibiliCrawler:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.img_key: Optional[str] = None
        self.sub_key: Optional[str] = None
        self._wbi_expire: float = 0
        self.crawled_bvids: set = set()
        self.crawled_uids: set = set()

    # ----------------------------------------------------------
    # Session 管理
    # ----------------------------------------------------------
    async def init_session(self):
        connector = aiohttp.TCPConnector(
            limit=MAX_CONCURRENT * 2,
            ssl=False,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        )

    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()
        await asyncio.sleep(0.25)

    # ----------------------------------------------------------
    # 公共请求头（每次随机UA + Cookie）
    # ----------------------------------------------------------
    def _headers(self) -> dict:
        headers = {
            "User-Agent":      random.choice(UA_POOL),
            "Referer":         "https://www.bilibili.com/",
            "Accept":          "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Origin":          "https://www.bilibili.com",
        }
        # 只有当 COOKIE 配置不为空时才添加到请求头中
        if COOKIE:
            headers["Cookie"] = COOKIE
        return headers

    # ----------------------------------------------------------
    # WBI 密钥刷新（每小时）
    # ----------------------------------------------------------
    async def _refresh_wbi(self):
        if time.time() < self._wbi_expire and self.img_key:
            return
        try:
            url = "https://api.bilibili.com/x/web-interface/nav"
            async with self.session.get(url, headers=self._headers()) as resp:
                if resp.status != 200:
                    raise ValueError(f"nav HTTP {resp.status}")
                data = await resp.json(content_type=None)
                wbi_img = data["data"]["wbi_img"]
                self.img_key = wbi_img["img_url"].rsplit("/", 1)[-1].split(".")[0]
                self.sub_key = wbi_img["sub_url"].rsplit("/", 1)[-1].split(".")[0]
                self._wbi_expire = time.time() + 3600
                logger.info("[OK] WBI 密钥刷新成功")
        except Exception as e:
            logger.warning(f"[WARN]  WBI 密钥刷新失败({e})，将使用无签名降级模式")
            self.img_key = None
            self.sub_key = None

    # ----------------------------------------------------------
    # 带重试的 GET 请求（核心防风控逻辑）
    # ----------------------------------------------------------
    async def _get(self, url: str, params: dict) -> Optional[dict]:
        """
        带指数退避重试的 GET 请求。
        处理：412风控、返回HTML、超时、JSON错误等所有异常。
        """
        for attempt in range(MAX_RETRY):
            # 随机请求间隔
            await asyncio.sleep(random.uniform(*REQ_INTERVAL))
            try:
                async with self.semaphore:
                    async with self.session.get(
                        url, params=params, headers=self._headers()
                    ) as resp:

                        # ---- HTTP 412: 触发风控 ----
                        if resp.status == 412:
                            wait = RETRY_BASE_SEC * (2 ** attempt) + random.uniform(3, 10)
                            logger.warning(
                                f"[BLOCK] HTTP 412 风控！第{attempt+1}/{MAX_RETRY}次重试，"
                                f"等待 {wait:.0f}s … {url}"
                            )
                            await asyncio.sleep(wait)
                            continue

                        # ---- 其他非200 ----
                        if resp.status != 200:
                            logger.warning(f"HTTP {resp.status}，URL: {url}")
                            await asyncio.sleep(5 * (attempt + 1))
                            continue

                        text = await resp.text(encoding="utf-8", errors="replace")

                        # ---- 返回了HTML（隐性风控）----
                        if text.lstrip().startswith("<!"):
                            wait = RETRY_BASE_SEC * (2 ** attempt) + random.uniform(3, 8)
                            logger.warning(
                                f"[BLOCK] 返回HTML（隐性风控）！"
                                f"第{attempt+1}/{MAX_RETRY}次重试，等待 {wait:.0f}s …"
                            )
                            await asyncio.sleep(wait)
                            continue

                        # ---- 解析JSON ----
                        try:
                            data = json.loads(text)
                        except json.JSONDecodeError:
                            logger.warning(f"JSON解析失败，URL: {url}")
                            await asyncio.sleep(5)
                            continue

                        # ---- API 返回 -412 ----
                        if data.get("code") == -412:
                            wait = RETRY_BASE_SEC * (2 ** attempt) + random.uniform(5, 15)
                            logger.warning(
                                f"[BLOCK] API -412 风控！"
                                f"第{attempt+1}/{MAX_RETRY}次重试，等待 {wait:.0f}s …"
                            )
                            await asyncio.sleep(wait)
                            continue

                        return data

            except asyncio.TimeoutError:
                wait = 5 * (attempt + 1)
                logger.warning(f"[TIMEOUT] 请求超时，第{attempt+1}次重试，等待{wait}s …")
                await asyncio.sleep(wait)
            except aiohttp.ClientConnectionError as e:
                wait = 8 * (attempt + 1)
                logger.warning(f"连接错误: {e}，等待{wait}s …")
                await asyncio.sleep(wait)
            except Exception as e:
                logger.warning(f"未知请求异常: {e}，第{attempt+1}次 …")
                await asyncio.sleep(5)

        logger.error(f"[X] 已达最大重试次数，放弃请求: {url} params={params}")
        return None

    # ----------------------------------------------------------
    # 获取UP主视频列表（分页 + 早停）
    # ----------------------------------------------------------
    async def _get_video_list(self, uid: int) -> list:
        """
        分页拉取UP主全部投稿，同时做时间早停：
        当页面最后一条视频早于2025年时，停止继续翻页。
        """
        all_vlist = []
        page = 1

        while True:
            await self._refresh_wbi()
            base_params = {
                "mid":   uid,
                "ps":    PAGE_SIZE,
                "pn":    page,
                "order": "pubdate",
                "tid":   0,
            }

            # 优先 WBI 接口
            if self.img_key and self.sub_key:
                signed = wbi_sign(base_params.copy(), self.img_key, self.sub_key)
                data = await self._get(
                    "https://api.bilibili.com/x/space/wbi/arc/search", signed
                )
            else:
                data = None

            # 降级到旧接口
            if not data or data.get("code") != 0:
                data = await self._get(
                    "https://api.bilibili.com/x/space/arc/search", base_params
                )

            if not data:
                raise RuntimeError("所有接口均请求失败（网络或风控）")

            code = data.get("code", -1)
            if code != 0:
                msg = data.get("message", "未知错误")
                raise RuntimeError(f"API错误 code={code} msg={msg}")

            page_data = data.get("data", {})
            vlist     = page_data.get("list", {}).get("vlist", [])
            total     = page_data.get("page", {}).get("count", 0)

            if not vlist:
                break

            # 早停判断
            last_pub = vlist[-1].get("created", 0)

            for v in vlist:
                pub = v.get("created", 0)
                if START_TS <= pub <= END_TS:
                    all_vlist.append(v)

            # 最后一条已早于2025，后面必然更早，停止翻页
            if last_pub < START_TS:
                logger.debug(f"  早停：页{page}末尾视频 {last_pub} < START_TS")
                break

            if page * PAGE_SIZE >= total:
                break

            page += 1

        return all_vlist

    # ----------------------------------------------------------
    # 获取视频详情
    # ----------------------------------------------------------
    async def _get_video_detail(self, bvid: str) -> Optional[dict]:
        data = await self._get(
            "https://api.bilibili.com/x/web-interface/view",
            {"bvid": bvid},
        )
        if data and data.get("code") == 0:
            return data.get("data", {})
        return None

    # ----------------------------------------------------------
    # 获取视频标签
    # ----------------------------------------------------------
    async def _get_video_tags(self, bvid: str) -> list[str]:
        data = await self._get(
            "https://api.bilibili.com/x/tag/archive/tags",
            {"bvid": bvid},
        )
        if data and data.get("code") == 0:
            return [t.get("tag_name", "") for t in (data.get("data") or [])
                    if t.get("tag_name")]
        return []

    # ----------------------------------------------------------
    # 判断是否合集
    # ----------------------------------------------------------
    @staticmethod
    def _is_collection(title: str, videos_count: int) -> int:
        if videos_count > 1:
            return 1
        if COLLECTION_RE.search(title):
            return 1
        return 0

    @staticmethod
    def _ts2str(ts: int) -> str:
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ""

    # ----------------------------------------------------------
    # 断点续传：加载进度
    # ----------------------------------------------------------
    def load_progress(self):
        if Path(PROGRESS_FILE).exists():
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                prog = json.load(f)
            self.crawled_uids = set(int(x) for x in prog.get("crawled_uids", []))
        else:
            self.crawled_uids = set()

        if Path(OUTPUT_FILE).exists():
            try:
                df_ex = pd.read_csv(OUTPUT_FILE, usecols=["bvid"], encoding="utf-8-sig")
                self.crawled_bvids = set(df_ex["bvid"].dropna().tolist())
                logger.info(f"[RESUME] 断点续传：加载已有 bvid {len(self.crawled_bvids)} 条，"
                            f"已完成UID {len(self.crawled_uids)} 个")
            except Exception as e:
                logger.warning(f"读取已有CSV失败: {e}")
                self.crawled_bvids = set()

    def save_progress(self, uid: int):
        self.crawled_uids.add(uid)
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump({"crawled_uids": list(self.crawled_uids)}, f, ensure_ascii=False)

    # ----------------------------------------------------------
    # 保存结果到 CSV
    # ----------------------------------------------------------
    FIELDNAMES = [
        "uid", "up_name", "fans", "level", "domain", "bvid", "title",
        "cover_url", "pubdate", "duration", "view", "like", "coin",
        "favorite", "share", "danmaku", "reply", "tags", "description",
        "copyright", "is_collection",
    ]

    def save_rows(self, rows: list):
        if not rows:
            return
        file_exists = Path(OUTPUT_FILE).exists()
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def backup(self):
        if Path(OUTPUT_FILE).exists():
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = f"video_backup_{ts}.csv"
            shutil.copy2(OUTPUT_FILE, dst)
            logger.info(f"[SAVE] 已备份数据 → {dst}")

    # ----------------------------------------------------------
    # 爬取单个UP主
    # ----------------------------------------------------------
    async def crawl_up(self, uid: int, up_name: str, fans: int) -> list:
        logger.info(f"[START] 开始爬取 UP 主: {up_name}  (UID: {uid}，粉丝: {fans:,})")
        level = classify_level(fans)
        rows  = []

        try:
            vlist = await self._get_video_list(uid)
            logger.info(f"  {up_name} 2025年视频共 {len(vlist)} 条")

            for v in vlist:
                bvid = v.get("bvid", "")
                if not bvid:
                    continue
                if bvid in self.crawled_bvids:
                    logger.debug(f"  跳过已有 bvid: {bvid}")
                    continue

                # ---- 获取视频详情 ----
                detail = await self._get_video_detail(bvid)
                if not detail:
                    logger.warning(f"  [WARN]  视频详情获取失败，跳过: {bvid}")
                    continue

                pub_ts = detail.get("pubdate", v.get("created", 0))
                # 二次确认时间范围
                if not (START_TS <= pub_ts <= END_TS):
                    continue

                stat  = detail.get("stat", {})
                title = detail.get("title", "")
                tid   = detail.get("tid", 0)

                # ---- 获取标签 ----
                tags = await self._get_video_tags(bvid)

                row = {
                    "uid":          uid,
                    "up_name":      up_name,
                    "fans":         fans,
                    "level":        level,
                    "domain":       TID_DOMAIN_MAP.get(tid, "其他"),
                    "bvid":         bvid,
                    "title":        title,
                    "cover_url":    detail.get("pic", ""),
                    "pubdate":      self._ts2str(pub_ts),
                    "duration":     detail.get("duration", 0),
                    "view":         stat.get("view", 0),
                    "like":         stat.get("like", 0),
                    "coin":         stat.get("coin", 0),
                    "favorite":     stat.get("favorite", 0),
                    "share":        stat.get("share", 0),
                    "danmaku":      stat.get("danmaku", 0),
                    "reply":        stat.get("reply", 0),
                    "tags":         ",".join(tags),
                    "description":  detail.get("desc", "").replace("\n", " ").replace("\r", ""),
                    "copyright":    detail.get("copyright", 0),
                    "is_collection": self._is_collection(title, detail.get("videos", 1)),
                }
                rows.append(row)
                self.crawled_bvids.add(bvid)

        except Exception as e:
            logger.error(
                f"[X] UP主 {up_name} (UID:{uid}) 爬取过程异常: {e}",
                exc_info=True,
            )

        logger.info(
            f"[OK] UP主 {up_name} 完成，本次获取 {len(rows)} 条，"
            f"等待 {UP_INTERVAL}s …"
        )
        return rows

    # ----------------------------------------------------------
    # 主流程
    # ----------------------------------------------------------
    async def run(self):
        # ---- 读取输入Excel ----
        logger.info(f"[READ] 读取输入文件: {INPUT_FILE}")
        df = pd.read_excel(INPUT_FILE, dtype=str)
        df.columns = df.columns.str.strip()

        # 自动识别列名（兼容多种写法）
        def find_col(df, candidates):
            for c in df.columns:
                for kw in candidates:
                    if kw.lower() in c.lower():
                        return c
            return None

        col_name = find_col(df, ["昵称", "名称", "name"])
        col_fans = find_col(df, ["粉丝", "fans"])
        col_uid  = find_col(df, ["uid", "UID"])

        if not all([col_name, col_fans, col_uid]):
            raise ValueError(
                f"无法识别输入列名！检测到列: {list(df.columns)}\n"
                "请确保有 昵称/粉丝量/UID 三列。"
            )

        logger.info(f"列映射: 昵称={col_name}, 粉丝量={col_fans}, UID={col_uid}")

        self.load_progress()

        await self.init_session()
        try:
            await self._refresh_wbi()

            up_done_count = 0

            for _, row in df.iterrows():
                uid     = int(str(row[col_uid]).strip())
                up_name = str(row[col_name]).strip()
                fans    = int(str(row[col_fans]).strip().replace(",", ""))

                if uid in self.crawled_uids:
                    logger.info(f"[SKIP] 跳过已完成 UP 主: {up_name} (UID:{uid})")
                    continue

                rows = await self.crawl_up(uid, up_name, fans)

                if rows:
                    self.save_rows(rows)

                self.save_progress(uid)
                up_done_count += 1

                # 每5个UP主备份
                if up_done_count % BACKUP_EVERY == 0:
                    self.backup()

                await asyncio.sleep(UP_INTERVAL)

        finally:
            await self.close_session()

        logger.info("[DONE] 全部UP主爬取完毕！")
        self.backup()


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    print(
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        "║   B站UP主2025年视频数据爬取脚本                      ║\n"
        "║   支持 WBI签名 / 412防风控 / 断点续传 / 自动备份     ║\n"
        "╚══════════════════════════════════════════════════════╝\n"
    )
    asyncio.run(BilibiliCrawler().run())