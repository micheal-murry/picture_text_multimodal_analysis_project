# -*- coding: utf-8 -*-
"""端到端分析管道编排器。

用法（在仓库根目录执行）：
    python run_pipeline.py --all             # 运行全部 10 步（步骤4需全量封面与CV依赖）
    python run_pipeline.py --steps 1 2 3     # 只运行指定步骤
    python run_pipeline.py --analysis        # 跳过采集重的封面特征步(4)，跑其余步骤
    python run_pipeline.py --sql             # 重建 SQLite 分析数据库并执行全部 SQL

说明：
    - 步骤1-5 为"数据工程"：清洗 -> 指标 -> 文本特征 -> 图像特征 -> 双模态特征
    - 步骤6-10 为"统计建模"：描述统计 -> 相关性/VIF -> 递进回归 -> 异质性 -> 稳健性
    - 步骤4 依赖全量封面（约 1GB，仓库内仅含12张样例）与 OpenCV/PaddleOCR 等可选依赖，
      仓库已提交其产出 image_features.csv / ocr_features.csv，复现可直接跳过。
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEP_DIR = ROOT / "src" / "analysis"

STEPS = {
    1:  "step01_cleaning.py",
    2:  "step02_propagation_metrics.py",
    3:  "step03_title_features.py",
    4:  "step04_image_features.py",
    5:  "step05_multimodal_features.py",
    6:  "step06_descriptive_analysis.py",
    7:  "step07_correlation_vif.py",
    8:  "step08_regression_modeling.py",
    9:  "step09_heterogeneity.py",
    10: "step10_robustness.py",
}

# 不需要全量封面 / 重CV依赖即可运行的分析步骤
ANALYSIS_STEPS = [1, 2, 3, 5, 6, 7, 8, 9, 10]


def run_step(n: int) -> bool:
    script = STEP_DIR / STEPS[n]
    if not script.exists():
        print(f"[缺失] 找不到 {script}")
        return False
    print("\n" + "=" * 72)
    print(f"[运行中] Step {n}: {STEPS[n]}")
    print("=" * 72)
    t0 = time.time()
    # 所有步骤均以仓库根目录为工作目录（脚本内部使用 outputs/ 与 data/ 相对路径）
    proc = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[失败] Step {n} 退出码 {proc.returncode}，耗时 {dt:.1f}s")
        return False
    print(f"[完成] Step {n}，耗时 {dt:.1f}s")
    return True


def build_sql() -> bool:
    print("\n" + "=" * 72)
    print("[SQL] 构建 SQLite 数据库并执行分析查询")
    print("=" * 72)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "sql" / "build_database.py")], cwd=str(ROOT)
    )
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser(description="B站知识区视频双模态分析管道")
    ap.add_argument("--all", action="store_true", help="运行全部 10 个步骤")
    ap.add_argument("--analysis", action="store_true",
                    help="运行分析步骤（跳过依赖全量封面的步骤4）")
    ap.add_argument("--steps", type=int, nargs="+", help="指定步骤编号，如 --steps 1 2")
    ap.add_argument("--sql", action="store_true", help="构建 SQLite 数据库并执行全部 SQL")
    args = ap.parse_args()

    if args.sql:
        ok = build_sql()
        sys.exit(0 if ok else 1)

    if args.all:
        todo = list(range(1, 11))
    elif args.analysis:
        todo = ANALYSIS_STEPS
    elif args.steps:
        todo = sorted(set(args.steps))
    else:
        ap.print_help()
        sys.exit(0)

    for n in todo:
        if n not in STEPS:
            print(f"[忽略] 未知步骤 {n}")
            continue
        if not run_step(n):
            sys.exit(1)

    print("\n全部选中步骤运行结束。")
    if args.all or args.analysis or args.steps:
        print("如需重建 SQL 分析库，请继续执行：python run_pipeline.py --sql")


if __name__ == "__main__":
    main()
