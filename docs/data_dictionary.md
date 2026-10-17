# 数据字典与指标口径

> 本文档对项目中的数据分层、字段口径、数据质量规则与数据字典做统一说明，
> 是本项目所有数据表/字段/指标的**唯一权威口径**，SQL、Notebook、看板均以此为准。

## 一、数据分层与文件清单

| 层级 | 文件 | 行数 | 生成步骤 | 说明 |
|---|---|---:|---|---|
| 原始层（raw） | `data/raw/video_full_data_2025.csv` | 4406 | 爬虫 | 采集到的全部视频原始记录（含后被剔除样本） |
| 原始层（raw） | `data/raw/data_clean_filtered.csv` | 4129 | 清洗前置 | 按领域/UP主筛选后的分析样本 |
| 维表层 | `data/raw/up_list_final.xlsx` | 68 | UP主采集 | 68 位 UP 主名单与领域、粉丝层级 |
| 清洗基表层 | `outputs/data_clean_base.csv` | 4129 | step01 | 去重/去缺失/异常标记后的干净基表 |
| 指标层 | `outputs/propagation_metrics.csv` | 4129 | step02 | 传播指标、互动率、对数化、综合指数 |
| 特征层 | `outputs/text_features.csv` | 4129 | step03 | 标题文本特征 |
| 特征层 | `outputs/image_features.csv` | 4129 | step04 | 封面视觉特征（仓库已提交成品，可跳过复现） |
| 特征层 | `outputs/ocr_features.csv` | 4129 | step04 | 封面 OCR 文字特征 |
| 特征层 | `outputs/multimodal_features.csv` | 4129 | step05 | 图文双模态特征、六类协同类型 |
| 宽表（建模用） | `outputs/modeling_dataset.csv` | 4129 | step05 合并 | 上述全表按 bvid 合并，共 133 列，Notebook/看板直接使用 |
| SQL 分析库 | `sql/bilibili_knowledge.db` | — | build_database.py | 7 张 SQLite 表（见第六节） |

主键：`bvid`（视频）；外键：`uid`（UP主）。所有特征表与指标表均以 `bvid` 一对一关联。

## 二、原始层与清洗基表字段

| 字段 | 中文名 | 类型 | 口径/取值 |
|---|---|---|---|
| uid | UP主ID | 标识 | B站用户唯一标识 |
| up_name | UP主昵称 | 字符串 | 账号名称 |
| fans | 粉丝数 | 整数 | 采集时 UP 主粉丝数量 |
| level | 粉丝层级 | 分类 | **-1=低层级，0=中层级，1=高层级**（按粉丝量分层，SQL 中需 CASE 映射） |
| domain | 知识领域 | 分类 | 科学科普 / 社科人文 / 财经商业 / 职场教育 |
| bvid | 视频BV号 | 标识 | 视频唯一主键 |
| title | 视频标题 | 文本 | 核心文本分析对象 |
| cover_url | 封面URL | 字符串 | 封面在线地址 |
| pubdate | 发布时间 | datetime | 格式 `YYYY-MM-DD HH:MM:SS`，样本均为 2025 年 |
| duration | 视频时长 | 整数（秒） | 研究样本最大值约 10 小时 |
| view/like/coin/favorite/share/danmaku/reply | 播放/点赞/投币/收藏/分享/弹幕/评论 | 整数 | 累计互动量 |
| tags | 标签 | 文本 | 逗号分隔 |
| description | 视频简介 | 文本 | 缺失填充为空字符串 |
| copyright | 版权类型 | 整数 | 1=原创，2=转载（另有3=其他口径） |
| is_collection | 是否合集 | 0/1 | 1=属于合集 |
| cover_exists | 封面是否存在 | 0/1 | 主分析仅保留 1 |
| flag_anomaly | 异常标记 | 0/1 | view/fans/duration 存在 0/负/极端值 |
| video_age_days | 上线天数 | 连续 | 采集日期 − 发布日期，控制曝光时长（采集基准日 2026-04-27） |
| log_age / log_fans / log_duration | 对数控制变量 | 连续 | 分别为 ln(age+1)、ln(fans+1)、ln(duration+1) |
| tag_count / description_len | 标签数 / 简介字符数 | 整数 | 控制变量 |

## 三、传播指标层口径

| 指标 | 公式 | 口径说明 |
|---|---|---|
| log_view | `ln(view+1)` | **核心因变量**，压缩播放量右偏长尾 |
| log_view_per_day | `ln(view/(video_age_days+1)+1)` | 控制曝光时长的传播速度，稳健性因变量 |
| recognition | `ln(like+coin+favorite+share+1)` | 认可类互动（对数） |
| participation | `ln(danmaku+reply+1)` | 讨论类参与（对数） |
| communication_effect | 三个标准化指标的均值 | 综合传播指数，仅作稳健性视角 |
| relative_view | `log_view − log_fans` | 去除粉丝基数后的相对传播力 |
| like_rate | `like/(view+1)` | 点赞率 |
| coin_rate | `coin/(view+1)` | 投币率（高成本认可，知识内容关键指标） |
| favorite_rate | `favorite/(view+1)` | 收藏率（内容留存价值） |
| share_rate | `share/(view+1)` | 分享率（二次传播） |
| comment_rate | `reply/(view+1)` | 评论率 |
| danmaku_rate | `danmaku/(view+1)` | 弹幕率 |
| top1_view / top1_fans | 播放/粉丝前 1% 标记 | 极端值仅标记、不删除 |
| high_spread_q75 | 日均播放前 25% 标记 | 高传播分组 |

**口径约定**：所有比率分母统一加 1（避免除零）；所有计数型变量在进入回归前做 `ln(x+1)` 对数化；
连续核心解释变量在进入模型前做 Z-score 标准化或中心化。

## 四、标题文本特征（step03）

| 字段 | 中文名 | 口径 |
|---|---|---|
| title_char_len / title_word_count | 标题字符长度 / 词数 | jieba 精确模式分词 |
| title_char_len_sq | 长度平方项 | 检验非线性 |
| title_question_type | 疑问式标题 | 含问号或疑问词（为什么/怎么/如何等） |
| title_exclamation_type | 感叹式标题 | 含感叹号 |
| title_has_number | 含数字 | 阿拉伯数字或中文数字（实验组分组字段） |
| title_number_count / title_number_phrase | 数字表达数量/短语 | 如"3分钟""10个" |
| curiosity_marker_count | 悬念词数量 | 37 词悬念词典命中数 |
| title_clickbait_score | 悬念表达强度 | 0.40×悬念词+0.25×疑问+0.20×感叹+0.15×情感（归一化） |
| practical_value_word_count | 实用价值词数量 | 35 词"干货/教程/方法"类词典 |
| title_sentiment_score | 标题情感得分 | [0,1]，>0.5 正向；优先 RoBERTa，未装 transformers 时降级为情感词典法 |
| title_sentiment_intensity | 情感极性强度 | `abs(score−0.5)`，越高越极端 |

## 五、封面与双模态特征

### 5.1 视觉特征（step04）

| 字段 | 口径 |
|---|---|
| image_brightness / image_contrast | 灰度均值 / 灰度标准差 |
| image_saturation / image_colorfulness | HSV 饱和度 / Hasler-Süsstrunk 色彩丰富度 |
| image_warm_cold | 暖冷色倾向（-1 冷 ~ +1 暖） |
| edge_density / image_entropy | Canny 边缘密度 / Shannon 灰度熵 |
| visual_complexity | 边缘密度、色彩丰富度、灰度熵等标准化后合成的视觉复杂度（0-1） |

### 5.2 OCR 与人物特征

| 字段 | 口径 |
|---|---|
| cover_has_text / cover_text_len / cover_text_box_count | 是否含文字 / 字符数 / 文字框数 |
| cover_text_area_ratio | 文字框面积占画面比例 |
| cover_text_sentiment_score | 封面 OCR 文字情感（SnowNLP） |
| human_presence / face_presence / face_count | 是否含人物 / 人脸（YOLO，MediaPipe/Haar 兜底） |

### 5.3 图文双模态特征（step05）

| 字段 | 中文名 | 口径 |
|---|---|---|
| image_title_fit | 图文语义匹配度 | 0.7×视觉代理余弦 + 0.3×OCR文字 TF-IDF 余弦，归一化[0,1]；CLIP 可用时替换为 CLIP 余弦（`clip_method_used` 记录实际方法） |
| image_title_fit_sq | 语义匹配度平方项 | 检验非线性 |
| title_cover_text_jaccard | 文字 Jaccard | 标题与封面 OCR 分词集合的交集/并集 |
| title_cover_text_tfidf_similarity | 文字 TF-IDF 余弦 | 字级 2-3gram |
| title_cover_text_overlap | 文字综合重合度 | Jaccard 与 TF-IDF 各自标准化后的均值 |
| sentiment_consistency / sentiment_conflict | 情感一致 / 情感冲突 | 标题与封面情感方向比较 |
| sentiment_intensity_gap | 情感强度差距 | 各自归一化后取绝对差 |
| semantic_text_interaction | 语义—文字交互项 | `image_title_fit × title_cover_text_overlap` |

### 5.4 六类双模态协同类型（complementarity_type）

| 编码 | 中文名 | 判定规则（P1→P6 优先级） | 样本量 | 平均播放对数 |
|---|---|---|---:|---:|
| complementary_explanation | 互补解释型 | 语义匹配中等、文字重叠低，信息互补 | 614 | 11.498 |
| low_relatedness | 低关联型 | 语义匹配低、文字重叠低 | 1430 | 11.233 |
| curiosity_contrast | 悬念反差型 | 语义中等、文字重叠中低 | 999 | 11.195 |
| emotional_conflict | 情感冲突型 | 标题与封面情感方向相反 | 213 | 11.104 |
| high_synergy | 高协同型 | 语义高匹配、文字高重叠 | 350 | 11.032 |
| redundant_reinforcement | 冗余强化型 | 语义高匹配且文字重叠最高（>0.4） | 523 | 10.360 |

> 判定逻辑可追溯：每条样本在 `synergy_rule_reason` 字段记录了命中的规则，便于审计复核。

## 六、SQL 分析库表结构（`sql/bilibili_knowledge.db`）

| 表 | 来源 | 粒度 |
|---|---|---|
| video_base | data_clean_base.csv | 视频 × 34 列（原始字段） |
| video_metrics | propagation_metrics.csv | 视频 × 56 列（指标） |
| text_features | text_features.csv | 视频 × 22 列 |
| image_features | image_features.csv | 视频 × 19 列 |
| ocr_features | ocr_features.csv | 视频 × 9 列 |
| multimodal | multimodal_features.csv | 视频 × 29 列 |
| up_master | 由视频表聚合 | UP主 × 68 行（video_count、总播放、领域、层级等） |

## 七、数据质量规则（15 项，全部通过）

见 `sql/01_data_quality_checks.sql`，包括：主键唯一性、跨表完整性、标题非空、播放量非负、
时长/发布时间范围、领域与层级枚举合法、互动率在 [0,1]、`log_view = ln(view+1)` 口径一致性、
UP主视频数聚合对账、封面缺失标记等。当前样本 15 项检查异常数均为 0。

## 八、数据合规说明

- 数据来自 B 站公开接口，仅包含**公开可见的元数据**（标题、封面、公开计数），不含任何用户隐私、评论内容或登录态信息；
- 爬虫代码中的登录 Cookie 已全部移除，改为从环境变量 `BILI_COOKIE` 读取（见 `.env.example`）；
- 约 1GB 全量封面图片不入库（`.gitignore` 排除），仓库仅保留 12 张样例用于演示；
- 数据仅限学术研究与教学交流，请勿用于商业用途或大规模二次分发。
