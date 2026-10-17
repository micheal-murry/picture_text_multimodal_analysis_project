-- =====================================================================
-- 01 数据质量校验报告
-- 对规范化后的分析库做入库后的常规数据质量校验。
-- 思路：每行一个检查项，bad_cnt 为违反规则的记录数，bad_cnt = 0 即通过。
-- 覆盖：主键唯一、跨表完整性、缺失值、数值范围、枚举合法性、口径对账。
-- =====================================================================

SELECT '1.主键唯一_bvid' AS check_item,
       COUNT(*) AS bad_cnt,
       'video_base 中 bvid 必须唯一' AS rule_desc
FROM (SELECT bvid FROM video_base GROUP BY bvid HAVING COUNT(*) > 1)

UNION ALL
SELECT '2.跨表完整性_指标表',
       COUNT(*),
       'video_base 每条视频在 video_metrics 中都应有一条记录'
FROM video_base b
LEFT JOIN video_metrics m ON b.bvid = m.bvid
WHERE m.bvid IS NULL

UNION ALL
SELECT '3.跨表完整性_文本特征',
       COUNT(*),
       'video_base 每条视频在 text_features 中都应有一条记录'
FROM video_base b
LEFT JOIN text_features t ON b.bvid = t.bvid
WHERE t.bvid IS NULL

UNION ALL
SELECT '4.跨表完整性_双模态特征',
       COUNT(*),
       'video_base 每条视频在 multimodal 中都应有一条记录'
FROM video_base b
LEFT JOIN multimodal mm ON b.bvid = mm.bvid
WHERE mm.bvid IS NULL

UNION ALL
SELECT '5.标题缺失',
       COUNT(*),
       'title 不应为空'
FROM video_base
WHERE title IS NULL OR length(trim(title)) = 0

UNION ALL
SELECT '6.播放量负值',
       COUNT(*),
       'view 应 >= 0'
FROM video_base
WHERE view < 0

UNION ALL
SELECT '7.时长异常',
       COUNT(*),
       'duration 应为正且 <= 36944 秒（约10小时，研究样本最大值）'
FROM video_base
WHERE duration <= 0 OR duration > 36944

UNION ALL
SELECT '8.发布时间超范围',
       COUNT(*),
       'pubdate 应落在 2025 年内'
FROM video_base
WHERE substr(pubdate, 1, 4) <> '2025'

UNION ALL
SELECT '9.知识领域枚举非法',
       COUNT(*),
       'domain 只能是四个研究领域之一'
FROM video_base
WHERE domain NOT IN ('科学科普', '社科人文', '财经商业', '职场教育')

UNION ALL
SELECT '10.UP层级枚举非法',
       COUNT(*),
       'level 只能取 -1/0/1（低/中/高）'
FROM video_base
WHERE level NOT IN (-1, 0, 1)

UNION ALL
SELECT '11.点赞率越界',
       COUNT(*),
       'like_rate 应在 [0,1] 区间'
FROM video_metrics
WHERE like_rate < 0 OR like_rate > 1

UNION ALL
SELECT '12.投币率越界',
       COUNT(*),
       'coin_rate 应在 [0,1] 区间'
FROM video_metrics
WHERE coin_rate < 0 OR coin_rate > 1

UNION ALL
SELECT '13.指标口径一致性',
       COUNT(*),
       'log_view 应等于 ln(view+1)'
FROM video_metrics
WHERE abs(log_view - (ln(view + 1))) > 1e-6

UNION ALL
SELECT '14.UP主视频数对账',
       COUNT(*),
       'up_master 聚合的 video_count 应与事实表一致'
FROM (
    SELECT b.uid, COUNT(*) AS c, COALESCE(u.video_count, -1) AS uc
    FROM video_base b
    LEFT JOIN up_master u ON u.uid = b.uid
    GROUP BY b.uid
)
WHERE c <> uc

UNION ALL
SELECT '15.封面缺失标记',
       COUNT(*),
       'cover_exists=0 表示缺封面，应为 0 条（清洗后样本）'
FROM video_base
WHERE cover_exists = 0

ORDER BY bad_cnt DESC;
