# -*- coding: utf-8 -*-
"""把 outputs/ 下的特征宽表构建为规范化 SQLite 分析库，并执行全部 SQL 查询。

运行：
    python sql/build_database.py

产物：
    sql/bilibili_knowledge.db            （已在 .gitignore，可随时重建）
    sql/query_results/*.csv              每个查询的结果
    sql/query_results/QUERY_RESULTS.md   汇总报告

数据库表设计（规范化，便于演示多表 JOIN）：
    up_master       UP主维度表（68 行）
    video_base      视频基础信息 + 清洗字段（4129 行）
    video_metrics   传播效果指标（4129 行）
    text_features   标题文本特征
    image_features  封面基础视觉特征
    ocr_features    封面 OCR 文字特征
    multimodal      标题-封面双模态协同特征
"""
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import paths  # noqa: E402

DB_PATH = paths.SQLITE_DB
SQL_DIR = ROOT / "sql"
RESULT_DIR = SQL_DIR / "query_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 表名 -> 数据文件（追加顺序遵循"先维度表后事实表"）
TABLES = [
    ("video_base",     paths.CLEAN_BASE_CSV),
    ("video_metrics",  paths.PROPAGATION_CSV),
    ("text_features",  paths.TEXT_FEATURES_CSV),
    ("image_features", paths.IMAGE_FEATURES_CSV),
    ("ocr_features",   paths.OCR_FEATURES_CSV),
    ("multimodal",     paths.MULTIMODAL_CSV),
]


def build_schema(conn: sqlite3.Connection) -> None:
    """从视频事实表聚合出 UP 主维度表，并建立常用索引。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS up_master AS
        SELECT
            uid,
            MIN(up_name)                                   AS up_name,
            MIN(fans)                                      AS fans,
            MIN(level)                                     AS level,
            MIN(domain)                                    AS domain,
            COUNT(*)                                       AS video_count,
            ROUND(AVG(view), 0)                            AS avg_views,
            SUM(view)                                      AS total_views
        FROM video_base
        GROUP BY uid
    """)
    for ddl in [
        "CREATE INDEX IF NOT EXISTS idx_video_base_uid  ON video_base(uid)",
        "CREATE INDEX IF NOT EXISTS idx_video_base_dom  ON video_base(domain)",
        "CREATE INDEX IF NOT EXISTS idx_video_base_bvid ON video_base(bvid)",
    ]:
        conn.execute(ddl)
    conn.commit()


def load_data(conn: sqlite3.Connection) -> None:
    for table, path in TABLES:
        if not path.exists():
            raise FileNotFoundError(f"找不到 {path}，请先运行分析步骤或确认 outputs/ 完整。")
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"[建表] {table:14s} {len(df):>5d} 行 × {df.shape[1]} 列  ← {path.name}")
    build_schema(conn)
    n_up = conn.execute("SELECT COUNT(*) FROM up_master").fetchone()[0]
    n_v = conn.execute("SELECT COUNT(*) FROM video_base").fetchone()[0]
    print(f"[建表] {'up_master':14s} {n_up:>5d} 行 （由视频表聚合生成）")
    print(f"数据库就绪：{n_up} 位 UP 主，{n_v} 条视频，路径 {DB_PATH}")


def split_statements(sql_text: str):
    """按分号切分语句，去掉注释行与空语句。"""
    clean = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        clean.append(line)
    text = "\n".join(clean)
    for stmt in re.split(r";\s*(?:\n|$)", text):
        stmt = stmt.strip()
        if stmt:
            yield stmt


def run_sql_files(conn: sqlite3.Connection) -> None:
    sql_files = sorted(SQL_DIR.glob("*.sql"))
    md_lines = ["# SQL 查询结果汇总\n",
                "> 由 `sql/build_database.py` 自动生成；数据库为 SQLite，零外部依赖。\n"]
    for sf in sql_files:
        md_lines.append(f"\n## {sf.name}\n")
        statements = list(split_statements(sf.read_text(encoding="utf-8")))
        q_idx = 0
        for stmt in statements:
            if not stmt.lower().startswith(("select", "with")):
                conn.execute(stmt)
                conn.commit()
                continue
            q_idx += 1
            tag = f"Q{q_idx}" if len(statements) > 1 else "Q1"
            try:
                df = pd.read_sql_query(stmt, conn)
            except Exception as e:
                print(f"[查询失败] {sf.name} {tag}: {e}")
                md_lines.append(f"### {tag} 查询失败\n\n```\n{e}\n```\n")
                continue
            suffix = f"_{q_idx}" if len(statements) > 1 else ""
            out_csv = RESULT_DIR / f"{sf.stem}{suffix}.csv"
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            preview = df.head(20).to_markdown(index=False)
            md_lines.append(f"### {tag}  返回 {len(df)} 行"
                            f"（`{out_csv.relative_to(ROOT)}`）\n")
            md_lines.append(preview + "\n")
            print(f"[查询] {sf.name} {tag}: {len(df)} 行 -> {out_csv.name}")
    (RESULT_DIR / "QUERY_RESULTS.md").write_text(
        "\n".join(md_lines), encoding="utf-8")
    print(f"\n汇总报告：{RESULT_DIR / 'QUERY_RESULTS.md'}")


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    try:
        load_data(conn)
        run_sql_files(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
