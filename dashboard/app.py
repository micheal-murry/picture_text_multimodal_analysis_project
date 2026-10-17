# -*- coding: utf-8 -*-
"""B站知识区视频"标题-封面"双模态传播指标看板（Streamlit）。

本地运行：
    streamlit run dashboard/app.py

页面结构：
    概览KPI / 月度趋势   -> 核心指标与时间趋势
    互动漏斗             -> 播放→点赞→投币→收藏→分享转化
    标题特征             -> 标题文本特征与播放量关系
    封面与双模态         -> 封面视觉特征与图文协同分析
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import paths  # noqa: E402

st.set_page_config(page_title="B站知识区双模态传播看板", layout="wide", page_icon=None)

# ----------------------------------------------------------------------
# 数据加载与预处理
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(paths.MODELING_CSV, encoding="utf-8-sig", low_memory=False)
    df["pubdate"] = pd.to_datetime(df["pubdate"], errors="coerce")
    df["pub_month"] = df["pubdate"].dt.strftime("%Y-%m")
    level_map = {1: "高层级", 0: "中层级", -1: "低层级"}
    df["level_name"] = df["level"].map(level_map).fillna("未知")
    synergy_map = {
        "high_synergy": "高协同型",
        "complementary_explanation": "互补解释型",
        "redundant_reinforcement": "冗余强化型",
        "curiosity_contrast": "悬念反差型",
        "emotional_conflict": "情感冲突型",
        "low_relatedness": "低关联型",
    }
    df["synergy_type"] = df["complementarity_type"].map(synergy_map)
    return df


st.title("B站知识区视频「标题-封面」双模态传播分析看板")
st.caption("样本：2025 年 68 位知识区 UP 主 · 4129 条视频 · 科学科普 / 社科人文 / 财经商业 / 职场教育")

with st.spinner("正在加载建模数据…"):
    df = load_data()

# ----------------------------------------------------------------------
# 全局筛选器
# ----------------------------------------------------------------------
st.sidebar.header("筛选")
domains = st.sidebar.multiselect("知识领域", sorted(df["domain"].unique()),
                                 default=sorted(df["domain"].unique()))
levels = st.sidebar.multiselect("UP主粉丝层级", ["低层级", "中层级", "高层级"],
                                default=["低层级", "中层级", "高层级"])
ups = st.sidebar.multiselect("UP主（可搜索，默认全部）",
                             sorted(df["up_name"].unique()))

d = df[df["domain"].isin(domains) & df["level_name"].isin(levels)].copy()
if ups:
    d = d[d["up_name"].isin(ups)]

if d.empty:
    st.warning("当前筛选条件下没有数据，请调整筛选器。")
    st.stop()

# ----------------------------------------------------------------------
# Tab 1：概览
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["① 概览", "② 互动漏斗", "③ 标题特征", "④ 封面与双模态"])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("视频数", f"{len(d):,}")
    c2.metric("UP主数", d["uid"].nunique())
    c3.metric("总播放量", f"{d['view'].sum()/1e8:.2f} 亿")
    c4.metric("中位播放量", f"{int(d['view'].median()):,}")
    c5.metric("条均播放量", f"{int(d['view'].mean()):,}")

    st.subheader("月度发布与播放趋势")
    m = (d.groupby("pub_month")
         .agg(发布视频数=("bvid", "count"),
              总播放量=("view", "sum"),
              平均播放量=("view", "mean"))
         .reset_index())
    m["平均播放量"] = m["平均播放量"].round(0)
    fig = px.bar(m, x="pub_month", y="发布视频数",
                 title="月度发布视频数（柱）与平均播放量（线）",
                 color_discrete_sequence=["#4C78A8"], labels={"pub_month": "月份"})
    fig.add_trace(go.Scatter(x=m["pub_month"], y=m["平均播放量"],
                             name="平均播放量", mode="lines+markers",
                             yaxis="y2", line=dict(color="#E45756", width=2)))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("各领域视频数占比")
        fig = px.pie(d, names="domain", hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, width="stretch")
    with cc2:
        st.subheader("各领域平均播放量（对数化）")
        g = d.groupby("domain", as_index=False)["log_view"].mean()
        fig = px.bar(g.sort_values("log_view"), x="log_view", y="domain",
                     orientation="h", color="domain",
                     labels={"log_view": "平均播放量对数", "domain": "领域"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------
# Tab 2：互动漏斗
# ----------------------------------------------------------------------
with tab2:
    st.subheader("内容互动漏斗（播放 → 点赞 → 投币 → 收藏 → 分享）")
    stages = [("播放", "view"), ("点赞", "like"), ("投币", "coin"),
              ("收藏", "favorite"), ("分享", "share"), ("弹幕", "danmaku"),
              ("评论", "reply")]
    funnel = pd.DataFrame({"环节": [s[0] for s in stages],
                           "数量": [int(d[s[1]].sum()) for s in stages]})
    fig = go.Figure(go.Funnel(
        y=funnel["环节"], x=funnel["数量"], textinfo="value+percent previous",
        marker=dict(color=px.colors.sequential.Blues_r[:7])))
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")

    st.subheader("各领域互动率对比（%）")
    rates = [("点赞率", "like_rate"), ("投币率", "coin_rate"),
             ("收藏率", "favorite_rate"), ("分享率", "share_rate"),
             ("评论率", "comment_rate"), ("弹幕率", "danmaku_rate")]
    rg = d.groupby("domain")[[c for _, c in rates]].mean().reset_index()
    rg = rg.melt(id_vars="domain", var_name="指标", value_name="数值")
    name_map = {c: n for n, c in rates}
    rg["指标"] = rg["指标"].map(name_map)
    rg["数值"] = (rg["数值"] * 100).round(2)
    fig = px.bar(rg, x="domain", y="数值", color="指标", barmode="group",
                 labels={"domain": "领域", "数值": "互动率（%）"})
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------
# Tab 3：标题特征
# ----------------------------------------------------------------------
with tab3:
    st.subheader("标题数字表达：含数字 vs 不含数字")
    g = (d.groupby("title_has_number")
         .agg(视频数=("bvid", "count"),
              平均播放量对数=("log_view", "mean"),
              平均日均播放对数=("log_view_per_day", "mean"),
              平均点赞率=("like_rate", "mean"))
         .reset_index())
    g["分组"] = g["title_has_number"].map({1: "含数字标题", 0: "不含数字标题"})
    g["平均点赞率"] = (g["平均点赞率"] * 100).round(2)
    g["平均播放量对数"] = g["平均播放量对数"].round(3)
    g["平均日均播放对数"] = g["平均日均播放对数"].round(3)
    st.dataframe(g[["分组", "视频数", "平均播放量对数", "平均日均播放对数", "平均点赞率"]],
                 width="stretch", hide_index=True)
    fig = px.bar(g, x="分组", y="平均播放量对数", color="分组",
                 range_y=[float(g["平均播放量对数"].min()) - 0.2,
                          float(g["平均播放量对数"].max()) + 0.15],
                 title="数字表达与播放量（对数均值）")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("标题策略指标分布（按组占比）")
    bins = pd.cut(d["title_char_len"], bins=[0, 12, 18, 24, 60],
                  labels=["≤12字", "13-18字", "19-24字", ">24字"])
    lg = d.groupby(bins, observed=False).agg(
        视频数=("bvid", "count"), 平均播放对数=("log_view", "mean")).reset_index()
    lg["title_char_len"] = lg["title_char_len"].astype(str)
    fig = px.bar(lg, x="title_char_len", y="平均播放对数",
                 color="视频数", title="标题长度分组的传播表现",
                 labels={"title_char_len": "标题长度", "平均播放对数": "平均播放量对数"})
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------
# Tab 4：封面与双模态
# ----------------------------------------------------------------------
with tab4:
    st.subheader("图文语义匹配度与传播效果")
    cc1, cc2 = st.columns(2)
    with cc1:
        fig = px.scatter(d.sample(min(1500, len(d)), random_state=42),
                         x="image_title_fit", y="log_view",
                         color="domain", opacity=0.55,
                         title="图文语义匹配度 vs 播放量对数（抽样1500）",
                         labels={"image_title_fit": "图文语义匹配度(CLIP余弦)",
                                 "log_view": "播放量对数"})
        st.plotly_chart(fig, width="stretch")
    with cc2:
        fig = px.scatter(d.sample(min(1500, len(d)), random_state=42),
                         x="title_cover_text_overlap", y="log_view",
                         color="domain", opacity=0.55,
                         trendline="lowess",
                         title="文字重叠度 vs 播放量对数（趋势线）",
                         labels={"title_cover_text_overlap": "标题-封面文字重叠度",
                                 "log_view": "播放量对数"})
        st.plotly_chart(fig, width="stretch")

    st.subheader("六类双模态协同类型传播表现")
    sg = (d.groupby("synergy_type")
          .agg(视频数=("bvid", "count"), 平均播放量对数=("log_view", "mean"),
               平均语义匹配度=("image_title_fit", "mean"),
               平均文字重叠度=("title_cover_text_overlap", "mean"))
          .reset_index().sort_values("平均播放量对数", ascending=False))
    for c in ["平均播放量对数", "平均语义匹配度", "平均文字重叠度"]:
        sg[c] = sg[c].round(3)
    st.dataframe(sg, width="stretch", hide_index=True)
    fig = px.bar(sg, x="synergy_type", y="平均播放量对数",
                 color="视频数", title="协同类型 × 平均播放量对数")
    st.plotly_chart(fig, width="stretch")

    st.info("核心结论：标题与封面文字的适度重合有利于传播（文字重叠度系数 0.383*），"
            "但当语义与文字层面同时高度重复时，传播效果反而下降（图文综合协同度系数 -0.642**）。"
            "有效入口设计是「主题一致、信息互补」，而非图文完全一致。")

st.caption("数据来源：B站公开接口 · 仅供学术研究 · 指标口径见 docs/data_dictionary.md")
