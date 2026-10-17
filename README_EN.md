# How Title–Cover Multimodal Synergy Affects Video Popularity: Evidence from Bilibili's Knowledge Zone

<p align="center">
  <b>Python collection & cleaning → text/image/multimodal feature engineering → SQL analytics database → sequential regression & A/B-test framework → Streamlit dashboard</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/sample-4129%20videos%20%C2%B7%2068%20creators-blue" />
  <img src="https://img.shields.io/badge/SQLite-7%20tables%20%C2%B7%2020%2B%20queries-green" />
  <img src="https://img.shields.io/badge/R%5E2-0.65-orange" />
  <img src="https://img.shields.io/badge/tests-16%20passed-success" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

A complete end-to-end data project from my undergraduate thesis:
**collection → cleaning → feature engineering → modeling → analysis → dashboard → data governance**.
It uses Bilibili's public API to study how video titles and cover images jointly affect view counts.
The full analysis is reproducible: all 16 unit tests and 15 SQL data-quality checks pass, and the
regression results match the thesis.

Methodology and metric definitions are documented in [`docs/`](docs/); Chinese README: [`README.md`](README.md).

## Key Findings

1. **Audience base is the dominant factor.** The coefficient of log followers is ≈1.09 (p<0.001);
   the top 5.7% of videos contribute 50% of all views.
2. **Numbered / list-style titles help.** After controlling for confounders, the coefficient of
   number-expression titles is 0.190 (p<0.10). A naive comparison shows the opposite sign because
   of category confounding — a **Simpson's paradox** (negative raw gap, positive after adjustment).
3. **Louder covers are not better covers.** Brightness, colorfulness, complexity, and face presence
   are not significant on their own.
4. **Titles and covers should complement each other rather than repeat each other.** Among the six
   synergy types, "complementary-explanatory" has the highest mean log views (11.50) and
   "redundant-reinforcing" the lowest (10.36); high duplication in both semantics and overlay text
   significantly suppresses popularity (interaction term -0.642, p<0.01).

## Tech Stack & Modules

| Module | Technology & implementation |
|---|---|
| Data collection | `src/scraper/`: async crawlers, polite rate limiting, cookie via environment variables, cover downloader |
| Data processing | `src/analysis/` 10-step pipeline with Pandas/NumPy (cleaning, feature engineering, modeling over 4,129 rows) |
| Statistical modeling | Sequential OLS, interaction effects, grouped heterogeneity (Chow tests), PSM / alternative outcomes / trimming robustness checks |
| Experiment analysis | `notebooks/03_ab_test.ipynb`: effect size, confidence intervals, Welch's t, Mann-Whitney, Cohen's d, MDE, confounding control |
| SQL analytics | `sql/` four SQL files: 15 quality checks, core metrics, window functions, topic analyses |
| Data governance | `docs/data_dictionary.md`: unified metric definitions, 15 data-quality rules, data layering |
| Dashboard | `dashboard/`: Streamlit app with four tabs (KPI / funnel / title / multimodal) |
| Topic analysis | `notebooks/04_business_recommendations.ipynb`, `docs/business_recommendations.md` |
| Testing | `tests/test_metrics.py`: 16 unit tests covering metrics and statistics |

## Repository Structure

```
bilibili-knowledge-multimodal-analysis/
├── data/raw/                 # Raw data (4,406 raw rows + 4,129 analysis rows + 68 creators)
│   └── NOTICE.md             # Data compliance notice
├── covers/                   # Cover samples (12 only; the full ~1 GB archive is not committed)
├── src/
│   ├── scraper/              # Collection: creator profiles, zone discovery, video crawling, cover download, classification
│   └── analysis/             # Analysis: step01 cleaning ~ step10 robustness
├── outputs/                  # Cleaned data, feature tables, regression results, figures (committed)
├── sql/                      # SQLite builder + 4 SQL query files + query results
├── notebooks/                # 4 notebooks with outputs (EDA / regression / A-B test / topic recommendations)
├── dashboard/app.py          # Interactive Streamlit dashboard
├── reports/                  # Selected thesis figures and analysis reports
├── docs/                     # Data dictionary, methodology, topic analysis
├── tests/test_metrics.py     # 16 unit tests
├── run_pipeline.py           # Pipeline orchestrator
└── requirements.txt
```

## Quick Start

```bash
# 1. Python 3.10–3.13 recommended
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Run the unit tests (16)
pytest -v

# 4. Reproduce the statistical analysis in one command
#    (skips the image-feature step that requires the 1 GB cover archive)
python run_pipeline.py --analysis

# 5. Rebuild the SQLite analytics database from scratch and run all SQL
python run_pipeline.py --sql

# 6. Run the interactive dashboard locally
streamlit run dashboard/app.py
```

- All processed feature tables (`outputs/*.csv`) are committed, so every analysis runs without
  downloading the cover images.
- To fully reproduce image features (OCR / face detection / CLIP) and RoBERTa sentiment, install
  `requirements-features.txt` and prepare the full cover archive (see `docs/methodology.md`, section 8).

## Analysis Pipeline (10 Steps)

| Phase | Step | Output |
|---|---|---|
| Data engineering | 1 Cleaning | data_clean_base.csv |
|  | 2 Propagation metrics | interaction rate, log transforms, composite index |
|  | 3 Title text features | jieba tokenization, number/question/teaser/utility words, sentiment |
|  | 4 Cover image features (needs archive) | color / complexity / OCR / faces |
|  | 5 Multimodal features | semantic matching, text overlap, synergy types |
| Statistical modeling | 6 Descriptive stats & difference tests | high- vs low-spread groups |
|  | 7 Correlation / VIF | multicollinearity checks |
|  | 8 Sequential regression | 4 models with creator-clustered robust SE |
|  | 9 Heterogeneity | grouped regressions + Chow tests |
|  | 10 Robustness | alternative outcomes / PSM / trimming |

## SQL Analytics Database

`python run_pipeline.py --sql` builds a self-contained SQLite database (7 tables) and runs:

- `01_data_quality_checks.sql`: **15 quality checks** — primary-key uniqueness, cross-table integrity,
  valid enums, metric reconciliation;
- `02_core_metrics.sql`: core metrics by category / spread level / month, creator rankings, interaction funnel;
- `03_window_functions.sql`: `ROW_NUMBER` / `RANK` / `LAG` / `NTILE` / cumulative distribution (Pareto);
- `04_title_cover_analysis.sql`: numbered-title grouping, six synergy types, Q4 vs Q1 comparison.

Query results are committed under `sql/query_results/`.

## Data Compliance

- Only **publicly visible metadata** (titles, covers, public counts) was collected — no user privacy,
  comments, or login-only data.
- All login cookies have been removed from the crawlers and are read from the environment variable
  `BILI_COOKIE` instead (see `.env.example`).
- Polite rate limiting is enforced (concurrency ≤ 3, 1.5–3.5 s delays, exponential backoff).
  Data is for academic research and educational use only.

## Author & License

- Author: mldl (Zhejiang University)
- Thesis advisor: Xi Chen
- Code released under the MIT License. Data copyright belongs to the platform and its creators;
  no commercial use or large-scale redistribution.
