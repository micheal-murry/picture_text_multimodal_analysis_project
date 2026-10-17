# B站知识区视频「标题-封面」双模态协同对传播效果的影响

<p align="center">
  <b>Python 采集 + 清洗 → 文本/图像/双模态特征工程 → SQL 指标库 → 递进回归 + A/B 实验框架 → Streamlit 看板</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/样本-4129%20条视频%20·%2068%20位UP主-blue" />
  <img src="https://img.shields.io/badge/SQLite-7%20张表%20·%2020%2B%20查询-green" />
  <img src="https://img.shields.io/badge/R%5E2-0.65-orange" />
  <img src="https://img.shields.io/badge/测试-16%20passed-success" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

本科毕业设计的完整数据项目，覆盖「采集 → 清洗 → 特征 → 建模 → 分析 → 看板 → 数据治理」的端到端流程。
数据来自 B 站公开接口，研究知识区视频的标题与封面如何共同影响播放量。分析过程可复现：
16 项单元测试与 15 项 SQL 质量校验全部通过，回归结果与论文一致。

方法与指标口径见 [`docs/`](docs/)，英文说明见 [`README_EN.md`](README_EN.md)。

## 一、核心发现

1. **账号基础盘是第一要素**：粉丝量对数系数 ≈1.09（p<0.001），5.7% 的视频贡献 50% 播放量；
2. **数字清单式标题有效**：「3个方法/5分钟看懂/10个误区」控制混杂后系数 0.190（p<0.10），
   且在朴素对比中因领域混杂呈现**辛普森悖论**（裸均值为负、控制后转正）；
3. **封面不是越醒目越好**：亮度、色彩、复杂度、人脸等单一视觉特征均不显著；
4. **图文要"信息互补"而非"标题复读"**：六类协同类型中「互补解释型」播放对数最高（11.50）、
   「冗余强化型」最低（10.36）；语义与文字双重高度重复显著抑制传播（交互项 -0.642，p<0.01）。

## 二、技术栈与模块

| 模块 | 技术与实现 |
|---|---|
| 数据采集 | `src/scraper/`：异步爬虫、礼貌限速、Cookie 环境变量化、封面下载 |
| 数据处理 | `src/analysis/` 十步分析管道，Pandas/NumPy 完成 4129 条数据的清洗、特征与建模 |
| 统计建模 | 递进式 OLS、交互效应、分组异质性（Chow 检验）、PSM/换因变量/去极端值稳健性检验 |
| 实验分析 | `notebooks/03_ab_test.ipynb`：效应量、置信区间、Welch t、Mann-Whitney、Cohen's d、MDE、混杂控制 |
| SQL 分析库 | `sql/` 下 4 个 SQL 文件：15 项质量校验、核心指标、窗口函数、专题分析 |
| 数据治理 | `docs/data_dictionary.md`：统一字段口径、15 项数据质量规则、数据分层 |
| 可视化看板 | `dashboard/`：Streamlit 四 Tab 看板（KPI/漏斗/标题/双模态） |
| 专题分析 | `notebooks/04_business_recommendations.ipynb`、`docs/business_recommendations.md` |
| 测试 | `tests/test_metrics.py`：16 项单元测试，覆盖指标计算与统计量 |

## 三、仓库结构

```
bilibili-knowledge-multimodal-analysis/
├── data/raw/                 # 原始数据（4406 原始样本 + 4129 分析样本 + 68 UP主名单）
│   └── NOTICE.md             # 数据合规声明
├── covers/                   # 封面样例（仅 12 张；全量约 1GB 不入库）
├── src/
│   ├── scraper/              # 采集：UP主画像、分区发现、视频抓取、封面下载、分类
│   └── analysis/             # 分析：step01 清洗 ~ step10 稳健性
├── outputs/                  # 清洗后数据、特征表、回归结果、图表（成品数据，已提交）
├── sql/                      # SQLite 建库 + 4 个分析查询文件 + 查询结果
├── notebooks/                # 4 个带输出的 Notebook（EDA/回归/AB实验/专题建议）
├── dashboard/app.py          # Streamlit 交互式看板
├── reports/                  # 论文精选图表与分析报告
├── docs/                     # 数据字典、方法论、专题分析
├── tests/test_metrics.py     # 16 项单元测试
├── run_pipeline.py           # 管道编排器
└── requirements.txt
```

## 四、快速开始

```bash
# 1. 建议 Python 3.10–3.13
python -m venv .venv && .venv\Scripts\activate      # Windows
# source .venv/bin/activate                          # macOS/Linux

# 2. 安装核心依赖
pip install -r requirements.txt

# 3. 运行单元测试（16 项）
pytest -v

# 4. 一条命令复现统计分析（跳过依赖 1GB 封面的图像特征步骤）
python run_pipeline.py --analysis

# 5. 从零重建 SQLite 分析库并执行全部 SQL
python run_pipeline.py --sql

# 6. 本地启动交互式看板
streamlit run dashboard/app.py
```

- 仓库**已提交全部成品特征数据**（`outputs/*.csv`），不下载封面也能完成所有分析；
- 若要从 0 复现封面特征（OCR/人脸/CLIP）与 RoBERTa 情感，再装 `requirements-features.txt`
  并准备全量封面归档（见 `docs/methodology.md` 第八节）。

## 五、数据分析步骤（10 步）

| 阶段 | 步骤 | 产出 |
|---|---|---|
| 数据工程 | 1 清洗 | data_clean_base.csv |
|  | 2 传播指标 | 互动率、对数化、综合指数 |
|  | 3 标题文本特征 | jieba 分词、数字/疑问/悬念/实用词、情感 |
|  | 4 封面图像特征（需封面归档） | 色彩/复杂度/OCR/人脸 |
|  | 5 双模态特征 | 语义匹配、文字重合、协同类型 |
| 统计建模 | 6 描述统计与差异检验 | 高低传播组对比 |
|  | 7 相关性/VIF | 多重共线性检验 |
|  | 8 递进回归 | 4 个模型 + uid 聚类稳健标准误 |
|  | 9 异质性 | 分组回归 + Chow 检验 |
|  | 10 稳健性 | 换因变量/PSM/去极端值 |

## 六、SQL 分析库

`python run_pipeline.py --sql` 生成零依赖的 SQLite 库（7 张表），并执行：

- `01_data_quality_checks.sql`：主键唯一、跨表完整、枚举合法、口径一致等 **15 项质量检查**；
- `02_core_metrics.sql`：领域/层级/月度核心指标、UP 主排行、互动漏斗；
- `03_window_functions.sql`：`ROW_NUMBER`/`RANK`/`LAG`/`NTILE`/累积分布（帕累托）等窗口函数；
- `04_title_cover_analysis.sql`：数字标题分组、六类协同类型、高/低传播 Q4 vs Q1 对比。

查询结果已随仓库提交于 `sql/query_results/`。

## 七、数据合规

- 仅采集 B 站**公开可见的元数据**（标题、封面、公开计数），不含用户隐私、评论或登录态；
- 爬虫中的登录 Cookie 已全部移除，改由环境变量 `BILI_COOKIE` 读取（见 `.env.example`）；
- 遵循礼貌限速（并发≤3、间隔 1.5–3.5 秒、失败退避），数据仅限学术研究与教学交流。

## 八、作者与许可

- 项目作者：mldl（zju）
- 指导教师：陈熹
- 代码以 MIT 协议开源；数据版权归原平台与创作者所有，请勿商用或大规模再分发。
