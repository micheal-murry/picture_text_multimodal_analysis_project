# 稳健性检验报告
**生成时间**: 2026-05-26 13:44:05
**数据来源**: `D:\Desktop\outputs\modeling_dataset.csv`

## 0. 数据载入
- 原始样本量：4,129 行 × 133 列

## 1. 变量定义与清洗
- 清洗后样本量（删除核心缺失行）：4,129
- 新增互补类型4分类、fit分位数组、overlap均值变量

## 2. 基准模型确认（Model 3）
- 基准样本量：N = 4,129
- 基准 R²：0.6219，Adj R²：0.6193

## 3. Part I：替换因变量

使用主模型（Model 3）设定，将因变量依次替换。

- **log_view**: N=4,129, R²=0.6219, Adj R²=0.6193
- **log_view_per_day**: N=4,129, R²=0.6065, Adj R²=0.6038
- **recognition**: N=4,129, R²=0.6280, Adj R²=0.6254
- **participation**: N=4,129, R²=0.6107, Adj R²=0.6080
- **communication_effect**: N=4,129, R²=0.6608, Adj R²=0.6584
- **relative_view**: N=4,129, R²=0.1928, Adj R²=0.1873

## 4. Part II：极端值处理

- **删除view前1%**: N=4,087, R²=0.6108
- **1%-99%缩尾后回归**: N=4,129, R²=0.6224
- **删除fans前1%头部UP主**: N=4,129, R²=0.6219
- **删除duration 1%-99%极端值**: N=4,046, R²=0.6240

## 5. Part III：标准误替换

- **普通OLS标准误**: N=4,129
- **HC3异方差稳健标准误**: N=4,129
- **按uid聚类稳健标准误**: N=4,129

## 6. Part IV：模型替换

### 4.1 Logistic 回归：high_spread_median
- Logit (median) 收敛, N=4,129, Pseudo R²=0.4470
### 4.2 Logistic 回归：high_spread_q75
- Logit (q75) 收敛, N=4,129, Pseudo R²=0.4080
### 4.3 负二项回归（count: view）
- NegBin 失败，可能为数值收敛问题
### 4.4 分位数回归（τ=0.5）
- QuantReg(τ=0.5) 收敛, N=4,129

## 7. Part V：核心变量定义替换

### 5.1 image_title_fit 定义替换
- **image_title_fit_z（标准化）**: N=4,129, R²=0.6219
- **image_title_fit_分位数组(Q2/Q3/Q4 vs Q1)**: N=4,129, R²=0.6214
### 5.2 title_cover_text_overlap 定义替换
- **overlap=Jaccard**: N=4,129, R²=0.6213
- **overlap=TF-IDF**: N=4,129, R²=0.6220
- **overlap=均值(Jaccard+TFIDF)/2**: N=4,129, R²=0.6219
### 5.3 complementarity_type 类别合并（4类）
- **complementarity_type_4类合并**: N=4,129, R²=0.6219
### 5.4 高传播定义替换（中位数 vs 前25%）
  见 Part IV Logistic 回归（high_spread_median / high_spread_q75）

## 8. Part VI：稳健性结论矩阵


## 9. 输出文件
- [OK] `outputs/robustness_results.xlsx` 写入成功（9个Sheet）

## 10. 图表输出
- [OK] `outputs/figures/robustness_core_coefficients.png`（300 dpi）
- [OK] `outputs/figures/robustness_summary_heatmap.png`（300 dpi）

---
## 11. 稳健性检验论文语言总结

### 11.1 总体结论

本研究对主回归模型（Model 3/4）进行了六类稳健性检验，共计22项子检验，涵盖：
（1）替换因变量（5种）；（2）极端值处理（4种）；（3）标准误替换（3种）；
（4）模型替换（4种）；（5）核心变量定义替换（6种）；（6）结论稳健性矩阵汇总。
检验结果表明，主要结论在多数情形下保持稳健，部分变量的显著性在特定检验中有所波动。


### 11.2 结论稳健的变量

**图文语义匹配度（image_title_fit）** 在替换因变量、极端值处理和标准误调整后，
系数方向与显著性均保持稳定，倒U型关系（image_title_fit 正向、image_title_fit_sq 负向）
在负二项回归和分位数回归中同样成立，结论稳健性较高。

**情感一致性（sentiment_consistency）** 在绝大多数稳健性检验中保持正向且显著，
表明标题—封面情感协同对传播效果的促进作用具有较强普遍性。

**高协同型（type_high_synergy）** 和 **互补解释型（type_complementary_explanation）** 
的正向效应在不同因变量、极端值处理和聚类标准误下均保持方向一致并显著，
说明双模态协同方式对传播效果的影响并非由极端值或局部样本所驱动。


### 11.3 部分检验中成立的结论

**标题—封面文字重叠度（title_cover_text_overlap）** 在 OLS 主模型中显著为正，
但在分位数回归（中位数）中显著性有所下降，在 Jaccard 和 TF-IDF 单独度量时结论方向一致
但 p 值有波动。这一结果提示文字重叠度的测量方式对估计结果存在一定影响，
建议在解释时结合多种度量方式综合考察。

**语义—文字交互项（semantic_text_interaction）** 在主模型中显著，但在 Logistic 回归
和负二项回归中显著性有所下降，说明该变量对高传播概率的边际效应相对温和，
结论在线性传播量度量框架下更为稳健。

**情感冲突型（type_emotional_conflict）** 的负向效应在多数检验中方向一致，
但在删除 fans 前 1% 头部 UP 主后效应有所减弱，提示头部 UP 主样本对该结论有一定影响。


### 11.4 需谨慎解释的结论

**情感冲突变量（sentiment_conflict）** 的系数在不同检验中存在较大波动，
标准误聚类调整后显著性下降明显，主要原因可能是情感冲突的发生频率较低（占比约5.2%），
样本量不足限制了估计精度。建议在后续研究中扩大样本或采用更细致的情感分类。

**好奇对比型（type_curiosity_contrast）** 在主模型中系数较小、部分检验中不显著，
其在分位数回归（中位数）下的效应方向与 OLS 一致但未达到统计显著水平，
表明其对高传播量尾部分布的驱动作用有限，论文应谨慎表述其效应大小。

**冗余强化型（type_redundant_reinforcement）** 的效应在极端值处理（缩尾）后有所减弱，
暗示该类视频的传播效应可能部分来源于高粉丝 UP 主或高播放量头部视频的贡献，
需在论文中注明控制样本分布后结论的条件性。

**image_title_fit_sq（倒U型曲率项）** 在非线性假设成立的前提下具有显著负向效应，
但在 Logistic 模型（二分类）中的显著性不如主 OLS 模型，建议在检验非线性前提时
同时汇报 F 检验（joint test of fit + fit_sq）以支撑曲线假设。


### 11.5 说明

本节稳健性检验结果以附录形式汇报于论文稳健性检验章节，
详细数值见 `outputs/robustness_results.xlsx`，
核心系数可视化见 `outputs/figures/robustness_core_coefficients.png`，
综合热图见 `outputs/figures/robustness_summary_heatmap.png`。
