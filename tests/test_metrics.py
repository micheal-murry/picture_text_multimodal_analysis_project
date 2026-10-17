# -*- coding: utf-8 -*-
"""核心指标与建模结果的单元测试。

运行（在仓库根目录）：
    pip install pytest
    pytest -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import paths  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(paths.MODELING_CSV, encoding="utf-8-sig", low_memory=False)


# ---------------------------------------------------------------- 数据规模
def test_sample_size(df):
    assert len(df) == 4129, "分析样本应为 4129 条视频"
    assert df["bvid"].nunique() == 4129, "bvid 应唯一"
    assert df["uid"].nunique() == 68, "应有 68 位 UP 主"


def test_domain_counts(df):
    counts = df["domain"].value_counts().to_dict()
    assert counts == {
        "社科人文": 1492,
        "科学科普": 1235,
        "财经商业": 811,
        "职场教育": 591,
    }


def test_level_counts(df):
    counts = df["level"].value_counts().to_dict()
    assert counts == {0: 1936, 1: 1083, -1: 1110}


# ---------------------------------------------------------------- 指标口径
def test_log_view_definition(df):
    expected = np.log(df["view"] + 1)
    assert np.allclose(df["log_view"], expected, atol=1e-6), "log_view 必须等于 ln(view+1)"


def test_relative_view_definition(df):
    assert np.allclose(df["relative_view"], df["log_view"] - df["log_fans"], atol=1e-6)


@pytest.mark.parametrize("col", [
    "like_rate", "coin_rate", "favorite_rate", "share_rate", "comment_rate", "danmaku_rate"
])
def test_rates_in_range(df, col):
    assert df[col].between(0, 1).all(), f"{col} 应落在 [0,1]"


def test_no_missing_core_fields(df):
    for col in ["bvid", "uid", "title", "view", "log_view", "domain", "level"]:
        assert df[col].notna().all(), f"{col} 不应有缺失"


# ---------------------------------------------------------------- 建模结论
def test_fans_coefficient_positive(df):
    m = smf.ols("log_view ~ log_fans + log_age + log_duration + C(domain)",
                data=df).fit(cov_type="cluster", cov_kwds={"groups": df["uid"]})
    assert m.params["log_fans"] > 0 and m.pvalues["log_fans"] < 0.001


def test_number_title_positive_after_controls(df):
    """数字标题控制混杂后系数应为正（论文 0.1904*，允许轻微数值波动）。"""
    f = (
        "log_view ~ log_fans + log_age + log_duration + tag_count + description_len "
        "+ is_collection + copyright + C(pub_month) + C(pub_weekday) + C(domain) + C(level) "
        "+ title_char_len + title_question_type + title_exclamation_type "
        "+ title_has_number + title_sentiment_intensity "
        "+ curiosity_marker_count + practical_value_word_count "
        "+ image_contrast + image_brightness + image_colorfulness "
        "+ visual_complexity + cover_has_text + cover_text_area_ratio + face_presence"
    )
    m = smf.ols(f, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["uid"]})
    assert m.params["title_has_number"] > 0.15
    assert m.rsquared > 0.62


def test_double_redundancy_interaction_negative(df):
    """语义匹配×文字重叠交互项应显著为负（核心结论）。"""
    for v in ["image_title_fit", "title_cover_text_overlap"]:
        df[v + "_z"] = (df[v] - df[v].mean()) / df[v].std(ddof=0)
    df["inter_z"] = df["image_title_fit_z"] * df["title_cover_text_overlap_z"]
    f = (
        "log_view ~ log_fans + log_age + log_duration + C(domain) + C(level) "
        "+ image_title_fit_z + title_cover_text_overlap_z + inter_z"
    )
    m = smf.ols(f, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["uid"]})
    assert m.params["inter_z"] < 0
    assert m.pvalues["inter_z"] < 0.05


# ---------------------------------------------------------------- SQL 层
def test_sqlite_tables():
    db = paths.SQLITE_DB
    if not db.exists():
        pytest.skip("SQLite 库尚未生成，请先运行 python run_pipeline.py --sql")
    import sqlite3
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    con.close()
    expected = {"video_base", "video_metrics", "text_features", "image_features",
                "ocr_features", "multimodal", "up_master"}
    assert expected <= tables
