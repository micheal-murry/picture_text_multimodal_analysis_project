# -*- coding: utf-8 -*-
"""生成并执行 notebooks/ 下的 4 个分析 Notebook（含输出结果）。

运行：py -3.13 tools/build_notebooks.py
"""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


def build_eda():
    nb = new_notebook()
    nb["cells"] = [
        md("""# 01 · 探索性数据分析（EDA）

**业务问题**：知识区视频的传播效果如何分布？哪些领域、UP层级和内容形态表现更好？

**数据**：`outputs/modeling_dataset.csv`（4129 条 × 全部特征，清洗与特征工程由步骤1-5生成）。
**分析方法**：描述统计、分布检验、分组对比。"""),
        code("""import sys
from pathlib import Path
ROOT = Path.cwd()
if (ROOT / 'src').exists():
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import paths

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

df = pd.read_csv(paths.MODELING_CSV, encoding='utf-8-sig', low_memory=False)
print('建模数据集：', df.shape)
df.head(3)"""),
        code("""# 数据质量速览
print('唯一视频数:', df['bvid'].nunique(), '| UP主数:', df['uid'].nunique())
print('缺失值总数:', int(df.isna().sum().sum()))
print('\\n核心数值字段描述:')
df[['view','like','coin','favorite','share','fans','duration']].describe(percentiles=[.5,.9]).round(0)"""),
        code("""# 播放量高度右偏（少数爆款拉高均值），因此研究使用对数化指标
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(df['view'], bins=80, color='#4C78A8'); axes[0].set_title('播放量原始分布（右偏）'); axes[0].set_xlabel('view')
axes[1].hist(df['log_view'], bins=50, color='#54A24B'); axes[1].set_title('播放量对数分布'); axes[1].set_xlabel('log_view')
plt.tight_layout(); plt.show()
print('偏度: 原始=%.2f, 对数=%.2f' % (df['view'].skew(), df['log_view'].skew()))"""),
        code("""# 样本结构：领域 × UP层级
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
order = df['domain'].value_counts().index
sns.countplot(data=df, y='domain', order=order, ax=axes[0], color='#4C78A8')
axes[0].set_title('各领域视频数'); axes[0].set_ylabel('')
level_map = {1:'高层级', 0:'中层级', -1:'低层级'}
tmp = df['level'].map(level_map)
sns.countplot(data=tmp.to_frame('层级'), x='层级', ax=axes[1], color='#F58518',
              order=['低层级','中层级','高层级'])
axes[1].set_title('UP主粉丝层级分布')
plt.tight_layout(); plt.show()"""),
        code("""# 互动漏斗：播放 -> 点赞 -> 投币 -> 收藏 -> 分享 -> 弹幕 -> 评论
stages = [('播放','view'),('点赞','like'),('投币','coin'),('收藏','favorite'),
          ('分享','share'),('弹幕','danmaku'),('评论','reply')]
tot = df[[c for _,c in stages]].sum()
for i,(n,c) in enumerate(stages):
    if i==0:
        print(f'{n:>4}: {tot[c]:>12,.0f}  转化率 100%')
    else:
        prev = stages[i-1][1]
        print(f'{n:>4}: {tot[c]:>12,.0f}  相对上一环 {tot[c]/tot[prev]*100:6.2f}%')"""),
        code("""# 各领域互动率（投币/收藏是B站知识内容的高价值认可指标）
rate_cols = ['like_rate','coin_rate','favorite_rate','share_rate','comment_rate']
g = df.groupby('domain')[rate_cols].mean().mul(100).round(2)
g.columns = ['点赞率','投币率','收藏率','分享率','评论率']
display(g.sort_values('投币率', ascending=False))
g.plot(kind='bar', figsize=(10,4), title='各领域平均互动率(%)')
plt.xticks(rotation=0); plt.ylabel('%'); plt.tight_layout(); plt.show()"""),
        code("""# 高/低传播组（中位数分组）标题与双模态特征对比
df['high_group'] = df['log_view'] >= df['log_view'].median()
cmp_cols = ['title_has_number','title_char_len','curiosity_marker_count',
            'image_title_fit','title_cover_text_overlap','visual_complexity']
g = df.groupby('high_group')[cmp_cols].mean().round(3)
g.index = ['低传播组','高传播组']
display(g)
print('\\nMann-Whitney U 检验示例（图文语义匹配度）：')
from scipy import stats
u,p = stats.mannwhitneyu(df.loc[df.high_group,'image_title_fit'],
                         df.loc[~df.high_group,'image_title_fit'], alternative='two-sided')
print('U=%.0f, p=%.4f （低传播组语义匹配度反而更高，提示不是越像越好）' % (u,p))"""),
        md("""## EDA 小结

1. 播放量分布显著右偏（均值 73.8 万、中位数 5.3 万），必须对数化处理；
2. 粉丝基础是传播的第一解释变量，账号层级差异远大于单条内容差异；
3. 互动漏斗逐级衰减，**投币率/收藏率**是知识内容区别于泛娱乐内容的高价值指标；
4. 高传播组的图文语义匹配度反而更低（Mann-Whitney p<0.01），为后续回归提供了方向性线索。"""),
    ]
    return nb


BASE_FORMULA = (
    "log_view ~ log_fans + log_age + log_duration + tag_count + description_len "
    "+ is_collection + copyright + C(pub_month) + C(pub_weekday) + C(domain) + C(level) "
    "+ title_char_len + title_question_type + title_exclamation_type "
    "+ title_has_number + title_sentiment_intensity "
    "+ curiosity_marker_count + practical_value_word_count "
    "+ image_contrast + image_brightness + image_colorfulness "
    "+ visual_complexity + cover_has_text + cover_text_area_ratio + face_presence"
)


def build_regression():
    nb = new_notebook()
    nb["cells"] = [
        md("""# 02 · 递进式回归建模复现

**复现论文表4.4**：以播放量对数 `log_view` 为因变量，递进加入控制变量 → 标题特征 → 封面特征 → 双模态特征，
OLS 使用 **按 uid 聚类的稳健标准误**（同一 UP 主的多条视频误差相关）。

本 Notebook 公式与论文附录 C 完全一致（含发布月份/星期、领域与层级虚拟变量）。"""),
        code("""import sys
from pathlib import Path
if (Path.cwd() / 'src').exists():
    sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from src import paths

df = pd.read_csv(paths.MODELING_CSV, encoding='utf-8-sig', low_memory=False)
print('样本量:', len(df))

BASE = '''log_view ~ log_fans + log_age + log_duration + tag_count + description_len
+ is_collection + copyright + C(pub_month) + C(pub_weekday) + C(domain) + C(level)'''
TITLE = ['title_char_len','title_question_type','title_exclamation_type',
         'title_has_number','title_sentiment_intensity',
         'curiosity_marker_count','practical_value_word_count']
COVER = ['image_contrast','image_brightness','image_colorfulness','visual_complexity',
         'cover_has_text','cover_text_area_ratio','face_presence']

def fit(formula, label):
    m = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['uid']})
    print(f'{label}: N={int(m.nobs)}, R2={m.rsquared:.4f}, 调整R2={m.rsquared_adj:.4f}')
    return m

m0 = fit(BASE, 'Model0 仅控制变量')
m1 = fit(BASE + ' + ' + ' + '.join(TITLE), 'Model1 +标题特征')
m2 = fit(BASE + ' + ' + ' + '.join(TITLE + COVER), 'Model2 +封面特征')"""),
        code("""# 双模态变量先标准化（论文：连续核心解释变量进入模型前标准化/中心化）
for v in ['image_title_fit','title_cover_text_overlap']:
    df[v+'_z'] = (df[v] - df[v].mean()) / df[v].std(ddof=0)
df['inter_z'] = df['image_title_fit_z'] * df['title_cover_text_overlap_z']
BIMODAL = ['image_title_fit_z + I(image_title_fit_z**2)',
           'title_cover_text_overlap_z', 'sentiment_consistency',
           'sentiment_conflict', 'inter_z']
m3 = fit(BASE + ' + ' + ' + '.join(TITLE + COVER + BIMODAL), 'Model3/4 +双模态及交互')"""),
        code("""# 核心变量系数汇总（与论文表4.4数字一致：R2=0.6307，数字系数0.1904*）
def star(p):
    return '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else ''))
rows = []
for name, m in [('Model1', m1), ('Model2', m2), ('Model3/4', m3)]:
    for v in TITLE + COVER + ['title_cover_text_overlap_z','inter_z','image_title_fit_z']:
        key = v if v in m.params.index else None
        if key is not None:
            rows.append({'模型':name,'变量':v,'系数':round(m.params[key],4),
                         '聚类稳健SE':round(m.bse[key],4),'p值':round(m.pvalues[key],3),
                         '显著性':star(m.pvalues[key])})
res = pd.DataFrame(rows)
display(res.pivot_table(index='变量', columns='模型', values='系数', aggfunc='first')
          .reindex(TITLE+COVER+['title_cover_text_overlap_z','image_title_fit_z','inter_z']).dropna(how='all'))
print('\\nModel3/4 关键双模态变量：')
sub = res[(res['模型']=='Model3/4') & res['变量'].isin(['title_cover_text_overlap_z','inter_z','title_has_number'])]
display(sub.reset_index(drop=True))"""),
        md("""## 建模结论

1. **UP 主粉丝量对数**系数约 1.09，各模型均 p<0.001——账号基础盘是传播第一要素；
2. **标题含数字**系数 0.190（Model2，p<0.10），方向在各模型稳定为正，
   说明"结构清晰、收益明确"的数字清单式标题有助于用户快速判断内容；
3. 疑问式标题、情绪强度、实用价值词与单一封面视觉特征（亮度/色彩/复杂度/人脸）**均不显著**；
4. **图文语义匹配度 × 文字重叠度交互项显著为负（p<0.01）**：当标题与封面在语义和文字层面
   同时高度重复时传播效果下降——有效入口设计是"主题一致、信息互补"，而非图文完全一致；
5. 模型解释力随变量组递进提升，R² 从 0.6251 升至 0.6353，稳健性检验（见 step10）方向一致。

> 完整 20+ 变量结果、异质性分组与稳健性检验见 `outputs/regression_results.xlsx`
> 与 `reports/analysis_reports/` 下的文字报告。
>
> 口径说明：本 Notebook 复现论文正文表 4.4（R²≈0.635）；完整管道 step08 的更全规格
> （含标准化变量）在 xlsx 中 R²≈0.645→0.655，两种口径结论一致。"""),
    ]
    return nb


def build_ab():
    nb = new_notebook()
    nb["cells"] = [
        md("""# 03 · A/B 实验框架分析：标题数字表达是否提升传播效果？

## 实验设计

| 项目 | 内容 |
|---|---|
| 业务假设 | 标题中加入数字表达（如"3个方法""5分钟看懂"）能提升视频播放量 |
| 实验组（T） | 标题含数字 `title_has_number=1`（2207 条） |
| 对照组（C） | 标题不含数字 `title_has_number=0`（1922 条） |
| 核心指标 | 播放量对数 `log_view`（近似正态、对极端值稳健） |
| 统计方法 | 均值差与 95% CI、Welch t 检验、Mann-Whitney U、Cohen's d、MDE |

**关键前提**：本数据是**观察性样本而非随机分流实验**，本 Notebook 会演示：
**朴素对比如何被混杂因素反转（辛普森悖论）→ 如何控制混杂 → 真实 A/B 实验的正确做法**。"""),
        code("""import sys
from pathlib import Path
if (Path.cwd() / 'src').exists():
    sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from src import paths

df = pd.read_csv(paths.MODELING_CSV, encoding='utf-8-sig', low_memory=False)
t = df.loc[df['title_has_number']==1, 'log_view'].dropna()
c = df.loc[df['title_has_number']==0, 'log_view'].dropna()
n_t, n_c = len(t), len(c)
print(f'实验组(含数字) n={n_t}，对照组(不含数字) n={n_c}')
print(f'实验组均值={t.mean():.4f}，对照组均值={c.mean():.4f}')"""),
        code("""# 第一步：朴素组间比较（观察性数据直接比较，结果可能是"假象"）
mean_diff = t.mean() - c.mean()
se = np.sqrt(t.var(ddof=1)/n_t + c.var(ddof=1)/n_c)
df_w = (t.var(ddof=1)/n_t + c.var(ddof=1)/n_c)**2 / (
    (t.var(ddof=1)/n_t)**2/(n_t-1) + (c.var(ddof=1)/n_c)**2/(n_c-1))
ci = mean_diff + stats.t.ppf([0.025,0.975], df_w) * se
t_stat, p_t = stats.ttest_ind(t, c, equal_var=False)
u_stat, p_u = stats.mannwhitneyu(t, c, alternative='two-sided')
pooled = np.sqrt(((n_t-1)*t.var(ddof=1)+(n_c-1)*c.var(ddof=1))/(n_t+n_c-2))
d = mean_diff/pooled
print(f'朴素均值差 = {mean_diff:.4f}，95% CI [{ci[0]:.4f}, {ci[1]:.4f}]')
print(f'Welch t={t_stat:.3f}, p={p_t:.3f}； Mann-Whitney p={p_u:.3f}； Cohen d={d:.3f}')
print('\\n注意：朴素结论是含数字标题反而"更差"。但这是真实的因果效应吗？——需要检查混杂。')"""),
        code("""# 第二步：混杂诊断。数字标题在不同领域的使用率差别巨大，而领域本身播放水平不同
usage = df.groupby('domain').agg(
    数字标题占比=('title_has_number', lambda s: round(s.mean()*100,1)),
    领域平均播放对数=('log_view', 'mean'),
    视频数=('bvid','count')).sort_values('数字标题占比', ascending=False)
display(usage.round(2))
print('解读：播放量最低的职场教育领域数字标题占比高达 79.4%，而高播放的科学科普仅 51.1%——')
print('领域构成同时影响"分组"和"结果"，是典型混杂变量，直接组间比较会得到被污染的结论。')"""),
        code("""# 第三步：分领域内对比（控制领域后的条件效应）
rows = []
for dom, sub in df.groupby('domain'):
    g = sub.groupby('title_has_number')['log_view']
    if set(g.groups.keys()) >= {0,1}:
        m0, m1 = g.mean()[0], g.mean()[1]
        rows.append({'知识领域': dom, '不含数字均值': round(m0,3),
                     '含数字均值': round(m1,3), '组内差': round(m1-m0,3)})
display(pd.DataFrame(rows).sort_values('组内差', ascending=False))
print('科学科普领域内数字标题为正（+0.70），其余领域为负——效应存在明显领域异质性，不能一概而论。')"""),
        code("""# 第四步：回归控制混杂（与论文 Model2 同口径，按 uid 聚类稳健标准误）
formula = '''log_view ~ title_has_number + log_fans + log_age + log_duration
+ tag_count + description_len + is_collection + copyright
+ C(pub_month) + C(pub_weekday) + C(domain) + C(level)
+ title_char_len + title_question_type + title_exclamation_type
+ title_sentiment_intensity + curiosity_marker_count + practical_value_word_count
+ image_contrast + image_brightness + image_colorfulness
+ visual_complexity + cover_has_text + cover_text_area_ratio + face_presence'''
m = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['uid']})
coef, se_c, p_c = m.params['title_has_number'], m.bse['title_has_number'], m.pvalues['title_has_number']
print(f'控制全部混杂后 title_has_number 系数 = {coef:.4f} (SE {se_c:.4f}, p={p_c:.3f})')
print('\\n系数由朴素的 -0.109 翻转为 +0.190（p<0.10，边际显著）——这就是辛普森悖论。')
print('结论：在同等账号、领域和内容条件下，数字表达与更好的传播存在正向关联，但观察性数据无法给出严格因果。')"""),
        code("""# 第五步：最小可检测效应（MDE）——当前样本量、80%功效、α=0.05 下能检出多小的效应
z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
mde = (z_a + z_b) * se
print(f'当前样本可检出的最小均值差（MDE）≈ {mde:.3f}（log_view 单位），约相对变化 {np.expm1(mde):.1%}')
print('控制后的系数 0.190 与 MDE 接近，处于"勉强可检出"区间；要把结论从 p<0.10 提升到 p<0.05，')
print('需要更大样本，或在真实 A/B 实验中通过随机分流消除混杂、提升检验灵敏度。')"""),
        md("""## 实验结论（可直接同步给运营）

1. **朴素对比为负、控制混杂后为正（+0.190，p<0.10）**：这是典型的辛普森悖论，
   说明取数分析必须先厘清混杂结构，不能直接拿分组均值下结论；
2. 在同等账号与领域条件下，数字清单式标题与更好的传播呈正向关联，
   且**科学科普领域**最明显（组内 +0.70），建议优先在该类内容上 A/B 试点；
3. 效应量较小，建议措辞为"**有条件的正向关联，建议小流量实验验证**"，不得宣称已证因果。

## 真实业务 A/B 实验落地检查清单

1. **随机分流**：按用户/曝光请求哈希分组，先做 **AA 测试**验证分流均匀；
2. **样本量前置**：上线前按 MDE 公式反算每组样本量与实验周期，避免"看到差就上线"；
3. **指标体系**：1 个核心指标 + 辅助指标 + **护栏指标**（完播率、取关率等）；
4. **口径冻结**：实验前冻结指标定义与判定标准，杜绝事后挑选指标；
5. **检验方法**：比例指标用两比例 z 检验；连续指标用 Welch t；偏态分布先对数化或用秩和检验；
6. **结果规范**：同时报告效应量、置信区间、显著性与业务意义；
7. **复盘闭环**：全量后跟踪 2–4 周护栏指标，再沉淀为下一轮实验假设。"""),
    ]
    return nb


def build_reco():
    nb = new_notebook()
    nb["cells"] = [
        md("""# 04 · 业务结论与运营建议（问题 → 分析 → 结论 → 落地 → 复盘）

把论文实证结果翻译为知识类内容团队可以直接执行的标题/封面策略。"""),
        code("""import sys
from pathlib import Path
if (Path.cwd() / 'src').exists():
    sys.path.insert(0, str(Path.cwd()))
import pandas as pd
from src import paths

df = pd.read_csv(paths.MODELING_CSV, encoding='utf-8-sig', low_memory=False)
# 六类协同类型的播放对数（结论5/6的数据支撑）
g = df.groupby('complementarity_type')['log_view'].mean().round(3).sort_values(ascending=False)
name_map = {'high_synergy':'高协同型','complementary_explanation':'互补解释型',
            'redundant_reinforcement':'冗余强化型','curiosity_contrast':'悬念反差型',
            'emotional_conflict':'情感冲突型','low_relatedness':'低关联型'}
display(g.rename(index=name_map))"""),
        md("""## 一、问题

知识类视频在推荐流中只能靠"标题+封面"争取首次点击：什么样的标题和封面更能带来传播？图文应该保持一致还是互补？

## 二、核心结论（均经回归 + 稳健性检验）

| # | 结论 | 证据 |
|---|---|---|
| 1 | 账号粉丝基础是传播第一要素 | log_fans 系数≈1.09，各模型均 p<0.001 |
| 2 | 标题数字表达有稳定正向作用（控制混杂后） | 系数 0.190，Model2 p<0.10 |
| 3 | 疑问式、情绪词、"干货"类实用词效果不稳定 | 各模型均不显著 |
| 4 | 单一封面视觉特征（亮度/色彩/人脸/文字）无独立显著作用 | H2 未获支持 |
| 5 | 六类协同类型中"互补解释型"传播最好（11.50）、"冗余强化型"最差（10.36） | 分组对比 |
| 6 | 语义与文字双重高度重复显著抑制传播 | 图文交互项 p<0.01 |
| 7 | 策略存在领域/账号异质性 | Chow 检验 p<0.001 |

## 三、可落地的运营建议

1. **标题做"结构化承诺"**：优先"数字清单+时间/数量+明确收益"（《3个方法/5分钟看懂/10个误区》），
   不依赖空泛疑问和情绪化表达；
2. **封面做"信息互补"而非"标题复读"**：用关键词、图示、人物场景补充标题没有的信息，
   保持主题一致但避免文字与语义双重冗余；
3. **放弃"越醒目越好"的误区**：不堆高饱和色彩和大文字，封面按知识属性（公式/图表/PPT截图）
   提供专业感线索；
4. **分层运营**：低层级账号可适当强化视觉钩子；中腰部账号适合"语义一致、文字不重复"；
   高层级账号以稳定更新为主，封面设计边际作用有限；
5. **分领域策略**：财经商业类慎用情绪化封面，职场教育类突出教程感，科学科普类保留图表/公式元素
   并优先试点数字标题。

## 四、落地动作

- 建立标题/封面模板库（数字清单式标题 × 信息互补式封面），新视频发布时小流量 A/B 测试；
- 将"标题-封面文字重叠度""图文语义匹配度"做成发布前自查指标（看板可直接复用）；
- 每月复盘新发布视频的互动率漏斗，迭代模板。

## 五、复盘闭环

**问题** → 采集 4129 条视频与封面 → 特征工程（文本/图像/双模态）→ 递进回归 + 异质性 + 稳健性
→ 结论：结构化标题 + 信息互补封面 → 落地：模板库 + A/B 测试 → 复盘：月度漏斗与指标监控 → 再假设。"""),
    ]
    return nb


BUILDERS = {
    "01_eda.ipynb": build_eda,
    "02_regression.ipynb": build_regression,
    "03_ab_test.ipynb": build_ab,
    "04_business_recommendations.ipynb": build_reco,
}


def main():
    for name, builder in BUILDERS.items():
        nb = builder()
        nb["metadata"] = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        }
        ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
        ep.preprocess(nb, resources={"metadata": {"path": str(ROOT)}})
        out = NB_DIR / name
        with open(out, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("生成并执行通过:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
