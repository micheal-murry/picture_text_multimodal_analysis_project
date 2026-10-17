# -*- coding: utf-8 -*-
"""统一的项目路径管理。

所有新增模块（SQL 构建、看板、notebook、测试）都从这里取路径，
避免在各处硬编码相对路径；原始 10 步分析脚本以仓库根目录为工作目录运行，
由 run_pipeline.py 统一调度。
"""
from pathlib import Path

# 仓库根目录：src/paths.py 向上一级
REPO_ROOT = Path(__file__).resolve().parents[1]

# 数据
RAW_DIR        = REPO_ROOT / "data" / "raw"
COVERS_DIR     = REPO_ROOT / "covers"
PROCESSED_DIR  = REPO_ROOT / "outputs"
FIGURES_DIR    = PROCESSED_DIR / "figures"
LOGS_DIR       = PROCESSED_DIR / "logs"

# 分析产物
REPORTS_DIR    = REPO_ROOT / "reports"
REPORT_FIG_DIR = REPORTS_DIR / "figures"

# 关键数据文件（与论文附录D一致）
RAW_CSV             = RAW_DIR / "video_full_data_2025.csv"
CLEAN_BASE_CSV      = PROCESSED_DIR / "data_clean_base.csv"
PROPAGATION_CSV     = PROCESSED_DIR / "propagation_metrics.csv"
TEXT_FEATURES_CSV   = PROCESSED_DIR / "text_features.csv"
IMAGE_FEATURES_CSV  = PROCESSED_DIR / "image_features.csv"
OCR_FEATURES_CSV    = PROCESSED_DIR / "ocr_features.csv"
MULTIMODAL_CSV      = PROCESSED_DIR / "multimodal_features.csv"
MODELING_CSV        = PROCESSED_DIR / "modeling_dataset.csv"

# SQL 数据库
SQLITE_DB = REPO_ROOT / "sql" / "bilibili_knowledge.db"


if __name__ == "__main__":
    for name, p in sorted(globals().items()):
        if isinstance(p, Path):
            print(f"{name:20s} {p}  {'[存在]' if p.exists() else '[缺失]'}")
