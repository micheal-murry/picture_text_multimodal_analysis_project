-- =====================================================================
-- 04 标题与封面双模态专题分析
-- 用 SQL 复现核心结论：数字标题分组、六类协同类型、高/低传播 Q4 vs Q1 对比。
-- =====================================================================

-- Q1 标题数字表达分组对比（论文 H1a：数字表达正向促进传播）
SELECT
    CASE t.title_has_number WHEN 1 THEN '标题含数字' ELSE '标题不含数字' END AS 分组,
    COUNT(*)                                                                AS 视频数,
    ROUND(AVG(b.view), 0)                                                   AS 平均播放量,
    ROUND(AVG(m.log_view), 3)                                               AS 平均播放量对数,
    ROUND(AVG(m.log_view_per_day), 3)                                       AS 平均日均播放对数,
    ROUND(AVG(m.like_rate) * 100, 2)                                        AS 平均点赞率_pct
FROM video_base b
JOIN video_metrics m ON b.bvid = m.bvid
JOIN text_features  t ON b.bvid = t.bvid
GROUP BY t.title_has_number
ORDER BY 平均播放量对数 DESC;

-- Q2 标题-封面文字重叠度高/低组对比（中位数分组；论文发现适度重合有助传播）
SELECT
    CASE WHEN mm.title_cover_text_overlap >=
              (SELECT title_cover_text_overlap FROM multimodal
               ORDER BY title_cover_text_overlap
               LIMIT 1 OFFSET (SELECT COUNT(*) FROM multimodal) / 2)
         THEN '高文字重叠' ELSE '低文字重叠' END AS 分组,
    COUNT(*)                                                                AS 视频数,
    ROUND(AVG(mm.title_cover_text_overlap), 3)                              AS 平均文字重叠度,
    ROUND(AVG(mm.image_title_fit), 3)                                       AS 平均语义匹配度,
    ROUND(AVG(m.log_view), 3)                                               AS 平均播放量对数
FROM video_base b
JOIN video_metrics m ON b.bvid = m.bvid
JOIN multimodal    mm ON b.bvid = mm.bvid
GROUP BY 分组
ORDER BY 平均播放量对数 DESC;

-- Q3 六类双模态协同类型的传播表现（论文表3.3 的分类体系）
SELECT
    CASE mm.complementarity_type
        WHEN 'high_synergy'              THEN '高协同型'
        WHEN 'complementary_explanation' THEN '互补解释型'
        WHEN 'redundant_reinforcement'   THEN '冗余强化型'
        WHEN 'curiosity_contrast'        THEN '悬念反差型'
        WHEN 'emotional_conflict'        THEN '情感冲突型'
        WHEN 'low_relatedness'           THEN '低关联型'
        ELSE mm.complementarity_type END                                     AS 协同类型,
    COUNT(*)                                                                AS 视频数,
    ROUND(AVG(b.view), 0)                                                    AS 平均播放量,
    ROUND(AVG(m.log_view), 3)                                                AS 平均播放量对数,
    ROUND(AVG(mm.image_title_fit), 3)                                        AS 平均语义匹配度,
    ROUND(AVG(mm.title_cover_text_overlap), 3)                               AS 平均文字重叠度
FROM video_base b
JOIN video_metrics m ON b.bvid = m.bvid
JOIN multimodal    mm ON b.bvid = mm.bvid
GROUP BY 协同类型
ORDER BY 平均播放量对数 DESC;

-- Q4 高传播组 vs 低传播组（按 log_view 四分位）的标题/封面特征对比
SELECT
    CASE WHEN q = 1 THEN '低传播Q1' WHEN 4 THEN '高传播Q4' ELSE '中间组' END AS 传播分组,
    COUNT(*)                                                                AS 视频数,
    ROUND(AVG(t.title_char_len), 1)                                         AS 平均标题字数,
    ROUND(AVG(t.title_question_type) * 100, 1)                              AS 疑问式标题占比_pct,
    ROUND(AVG(t.title_has_number) * 100, 1)                                 AS 数字标题占比_pct,
    ROUND(AVG(t.curiosity_marker_count), 3)                                 AS 悬念词均值,
    ROUND(AVG(mm.image_title_fit), 3)                                       AS 图文语义匹配度,
    ROUND(AVG(mm.title_cover_text_overlap), 3)                              AS 文字重叠度
FROM (
    SELECT bvid, NTILE(4) OVER (ORDER BY log_view) AS q
    FROM video_metrics
) x
JOIN video_base b ON b.bvid = x.bvid
JOIN text_features t ON t.bvid = x.bvid
JOIN multimodal   mm ON mm.bvid = x.bvid
WHERE x.q IN (1, 4)
GROUP BY x.q;
