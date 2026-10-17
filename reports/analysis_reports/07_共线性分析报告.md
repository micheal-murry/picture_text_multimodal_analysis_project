# 第七步：相关性与多重共线性检验报告

> 生成时间：2026-05-25 15:26:26  
> 样本量 N = 4129

---

## 一、数据质量

| 变量 | 问题 | 处理方式 |
|------|------|---------|
| `cover_has_chart_or_formula` | 严重缺失 100% | 已排除出分析 |
| `cover_is_slide_or_screenshot` | 严重缺失 100% | 已排除出分析 |

---

## 二、高度相关变量对（|r| ≥ 0.5）

| 变量A | 含义A | 变量B | 含义B | Pearson r |
|-------|-------|-------|-------|-----------|
| `log_view` | 对数总播放量 | `log_view_per_day` | 对数日均播放量 | 0.9870 |
| `recognition` | 认知传播效果 | `communication_effect` | 综合传播效果 | 0.9808 |
| `image_title_fit` | 图文语义匹配度 | `image_title_fit_sq` | 图文匹配度平方项 | 0.9751 |
| `title_cover_text_overlap` | 标封文字重叠度 | `semantic_text_interaction` | 语义—文字交互 | 0.9625 |
| `log_view` | 对数总播放量 | `recognition` | 认知传播效果 | 0.9595 |
| `log_view` | 对数总播放量 | `communication_effect` | 综合传播效果 | 0.9556 |
| `log_view_per_day` | 对数日均播放量 | `recognition` | 认知传播效果 | 0.9471 |
| `log_view_per_day` | 对数日均播放量 | `communication_effect` | 综合传播效果 | 0.9385 |
| `participation` | 参与传播效果 | `communication_effect` | 综合传播效果 | 0.9114 |
| `recognition` | 认知传播效果 | `participation` | 参与传播效果 | 0.8336 |
| `communication_effect` | 综合传播效果 | `log_fans` | 对数粉丝数 | 0.7739 |
| `log_view` | 对数总播放量 | `participation` | 参与传播效果 | 0.7619 |
| `recognition` | 认知传播效果 | `log_fans` | 对数粉丝数 | 0.7601 |
| `log_view_per_day` | 对数日均播放量 | `participation` | 参与传播效果 | 0.7387 |
| `log_view` | 对数总播放量 | `log_fans` | 对数粉丝数 | 0.7297 |
| `log_view_per_day` | 对数日均播放量 | `log_fans` | 对数粉丝数 | 0.7274 |
| `log_view` | 对数总播放量 | `relative_view` | 相对播放量 | 0.7169 |
| `participation` | 参与传播效果 | `log_fans` | 对数粉丝数 | 0.7142 |
| `log_view_per_day` | 对数日均播放量 | `relative_view` | 相对播放量 | 0.7003 |
| `image_title_fit_sq` | 图文匹配度平方项 | `semantic_text_interaction` | 语义—文字交互 | 0.6687 |
| `recognition` | 认知传播效果 | `relative_view` | 相对播放量 | 0.6267 |
| `image_title_fit` | 图文语义匹配度 | `semantic_text_interaction` | 语义—文字交互 | 0.6091 |
| `communication_effect` | 综合传播效果 | `relative_view` | 相对播放量 | 0.6070 |
| `image_title_fit_sq` | 图文匹配度平方项 | `title_cover_text_overlap` | 标封文字重叠度 | 0.5492 |
| `image_title_fit` | 图文语义匹配度 | `title_cover_text_overlap` | 标封文字重叠度 | 0.5163 |

---

## 三、极高相关变量对（|r| ≥ 0.7）

| 变量A | 变量B | r |
|-------|-------|---|
| `log_view` | `log_view_per_day` | 0.9870 |
| `recognition` | `communication_effect` | 0.9808 |
| `image_title_fit` | `image_title_fit_sq` | 0.9751 |
| `title_cover_text_overlap` | `semantic_text_interaction` | 0.9625 |
| `log_view` | `recognition` | 0.9595 |
| `log_view` | `communication_effect` | 0.9556 |
| `log_view_per_day` | `recognition` | 0.9471 |
| `log_view_per_day` | `communication_effect` | 0.9385 |
| `participation` | `communication_effect` | 0.9114 |
| `recognition` | `participation` | 0.8336 |
| `communication_effect` | `log_fans` | 0.7739 |
| `log_view` | `participation` | 0.7619 |
| `recognition` | `log_fans` | 0.7601 |
| `log_view_per_day` | `participation` | 0.7387 |
| `log_view` | `log_fans` | 0.7297 |
| `log_view_per_day` | `log_fans` | 0.7274 |
| `log_view` | `relative_view` | 0.7169 |
| `participation` | `log_fans` | 0.7142 |
| `log_view_per_day` | `relative_view` | 0.7003 |

---

## 四、VIF 多重共线性检验

| 变量（英文） | 中文含义 | VIF | 判断 |
|-------------|---------|-----|------|
| `semantic_text_interaction_z` | 语义—文字交互 | 36.048 | ✗ 严重共线 |
| `pub_month_12` | pub_month_12 | 29.159 | ✗ 严重共线 |
| `cover_has_text` | 封面含文字 | 26.961 | ✗ 严重共线 |
| `title_cover_text_overlap_z` | titleover_text_overlap | 26.381 | ✗ 严重共线 |
| `log_age_z` | 对数视频年龄（天） | 21.126 | ✗ 严重共线 |
| `level_1` | level_1 | 17.017 | ✗ 严重共线 |
| `pub_month_11` | pub_month_11 | 16.765 | ✗ 严重共线 |
| `level_0` | level_0 | 10.114 | ✗ 严重共线 |
| `pub_month_10` | pub_month_10 | 9.893 | [WARN] 需关注 |
| `log_fans_z` | 对数粉丝数 | 9.387 | [WARN] 需关注 |
| `pub_month_9` | pub_month_9 | 6.795 | [WARN] 需关注 |
| `pub_month_8` | pub_month_8 | 4.113 | ✓ 可接受 |
| `face_presence` | 封面含人脸 | 3.524 | ✓ 可接受 |
| `image_title_fit_c_z` | 图文语义匹配度 | 3.501 | ✓ 可接受 |
| `title_has_number` | 标题含数字 | 2.945 | ✓ 可接受 |
| `domain_财经商业` | domain_财经商业 | 2.878 | ✓ 可接受 |
| `title_question_type` | 标题疑问句型 | 2.801 | ✓ 可接受 |
| `domain_科学科普` | domain_科学科普 | 2.736 | ✓ 可接受 |
| `pub_month_7` | pub_month_7 | 2.677 | ✓ 可接受 |
| `cover_text_area_ratio_z` | 封面文字占比 | 2.606 | ✓ 可接受 |
| `image_title_fit_c_sq_z` | 图文匹配度平方项 | 2.529 | ✓ 可接受 |
| `pub_month_6` | pub_month_6 | 2.239 | ✓ 可接受 |
| `sentiment_consistency` | sentimentonsistency | 2.013 | ✓ 可接受 |
| `pub_month_5` | pub_month_5 | 2.0 | ✓ 可接受 |
| `log_duration_z` | 对数视频时长 | 1.981 | ✓ 可接受 |
| `title_char_len_z` | titlehar_len | 1.951 | ✓ 可接受 |
| `title_exclamation_type` | 标题感叹句型 | 1.934 | ✓ 可接受 |
| `domain_职场教育` | domain_职场教育 | 1.874 | ✓ 可接受 |
| `pub_month_3` | pub_month_3 | 1.833 | ✓ 可接受 |
| `pub_month_2` | pub_month_2 | 1.813 | ✓ 可接受 |
| `pub_month_4` | pub_month_4 | 1.78 | ✓ 可接受 |
| `tag_count_z` | tagount | 1.748 | ✓ 可接受 |
| `both_high_arousal` | 双高唤起 | 1.635 | ✓ 可接受 |
| `image_colorfulness_z` | imageolorfulness | 1.577 | ✓ 可接受 |
| `visual_complexity_z` | visualomplexity | 1.529 | ✓ 可接受 |
| `image_brightness_z` | 封面亮度 | 1.525 | ✓ 可接受 |
| `image_contrast_z` | imageontrast | 1.523 | ✓ 可接受 |
| `title_sentiment_intensity_z` | 标题情感强度 | 1.448 | ✓ 可接受 |
| `sentiment_conflict` | sentimentonflict | 1.326 | ✓ 可接受 |
| `description_len_z` | 简介字符数 | 1.307 | ✓ 可接受 |
| `curiosity_marker_count_z` | curiosity_markerount | 1.105 | ✓ 可接受 |
| `practical_value_word_count_z` | practical_value_wordount | 1.072 | ✓ 可接受 |

---

## 五、自动诊断与建议

### 5.1 高度相关变量说明

- **`image_title_fit` ↔ `image_title_fit_sq`（r=0.975）**：已对原变量进行均值中心化后重建平方项（`image_title_fit_c_sq`），中心化后共线性显著降低，VIF 见上表。
- **`recognition` ↔ `participation`（r=0.834）**：两者均为因变量，相关较高反映共同驱动因素，属正常现象；回归时分别作为因变量独立建模，无需删除。
- **`title_cover_text_overlap` ↔ `semantic_text_interaction`（r=0.963）**：标封文字重叠度 与 语义—文字交互 存在高度相关，在主模型中建议择一保留，另一变量进入稳健性模型。
- **`image_title_fit_sq` ↔ `semantic_text_interaction`（r=0.669）**：图文匹配度平方项 与 语义—文字交互 存在高度相关，在主模型中建议择一保留，另一变量进入稳健性模型。
- **`image_title_fit` ↔ `semantic_text_interaction`（r=0.609）**：图文语义匹配度 与 语义—文字交互 存在高度相关，在主模型中建议择一保留，另一变量进入稳健性模型。
- **`image_title_fit_sq` ↔ `title_cover_text_overlap`（r=0.549）**：图文匹配度平方项 与 标封文字重叠度 存在高度相关，在主模型中建议择一保留，另一变量进入稳健性模型。
- **`image_title_fit` ↔ `title_cover_text_overlap`（r=0.516）**：图文语义匹配度 与 标封文字重叠度 存在高度相关，在主模型中建议择一保留，另一变量进入稳健性模型。

### 5.2 VIF 严重共线（≥10）

- **`semantic_text_interaction_z`（VIF=36.048）**：存在严重多重共线性，建议从主模型中移除，或与其高度相关变量择一。
- **`pub_month_12`（VIF=29.159）**：存在严重多重共线性，建议从主模型中移除，或与其高度相关变量择一。
- **`cover_has_text`（VIF=26.961）**：存在严重多重共线性，建议从主模型中移除，或与其高度相关变量择一。
- **`title_cover_text_overlap_z`（VIF=26.381）**：存在严重多重共线性，建议从主模型中移除，或与其高度相关变量择一。
- **`log_age_z`（VIF=21.126）**：存在严重多重共线性，建议从主模型中移除，或与其高度相关变量择一。
- **`level_1`（VIF=17.017）**：存在严重多重共线性，建议从主模型中移除，或与其高度相关变量择一。
- **`pub_month_11`（VIF=16.765）**：存在严重多重共线性，建议从主模型中移除，或与其高度相关变量择一。
- **`level_0`（VIF=10.114）**：存在严重多重共线性，建议从主模型中移除，或与其高度相关变量择一。

### 5.3 需关注（5≤VIF<10）

- `pub_month_10`（VIF=9.893）：存在一定共线性，可保留但须在论文中说明并进行稳健性检验。
- `log_fans_z`（VIF=9.387）：存在一定共线性，可保留但须在论文中说明并进行稳健性检验。
- `pub_month_9`（VIF=6.795）：存在一定共线性，可保留但须在论文中说明并进行稳健性检验。

### 5.4 主回归模型建议保留变量


**控制变量：**
`log_fans`、`log_age`、`log_duration`、`tag_count`、`description_len`、
`domain` 哑变量、`pub_month` 哑变量

**标题特征：**
`title_char_len`（优先于 title_word_count）、`title_question_type`、
`title_exclamation_type`、`title_has_number`、`title_sentiment_intensity`、
`curiosity_marker_count`、`practical_value_word_count`

**封面特征：**
`image_brightness`、`image_colorfulness`、`visual_complexity`、
`cover_text_area_ratio`（优先于 cover_has_text）、`face_presence`

**双模态协同：**
`image_title_fit_c`（均值中心化）、`image_title_fit_c_sq`（中心化平方项）、
`title_cover_text_overlap`、`sentiment_consistency`、
`both_high_arousal`、`semantic_text_interaction`

### 5.5 适合放入稳健性或替代模型的变量


- `title_word_count`：`title_char_len` 的替代测量
- `cover_has_text`：`cover_text_area_ratio` 的二值替代
- `sentiment_conflict`：与 `sentiment_consistency` 互斥，用于对立假设模型
- `image_contrast`：与 `image_colorfulness` / `visual_complexity` 有一定重叠
- `log_view_per_day`、`recognition`、`participation`、`communication_effect`、
  `relative_view`：各维度因变量，分别作为主模型依变量进行敏感性分析


---

> *本报告由 Python 脚本自动生成，数据结论以实际输出为准。*