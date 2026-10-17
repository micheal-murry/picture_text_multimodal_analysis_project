-- =====================================================================
-- 02 核心指标聚合：多表 JOIN + GROUP BY
-- 从不同维度汇总核心传播指标，并给出 UP 主排行与互动漏斗。
-- =====================================================================

-- Q1 各知识领域传播总览：视频数、UP数、播放规模、互动率、标题数字表达占比
SELECT
    b.domain                                                AS 知识领域,
    COUNT(*)                                                AS 视频数,
    COUNT(DISTINCT b.uid)                                   AS UP主数,
    SUM(b.view)                                             AS 总播放量,
    ROUND(AVG(b.view), 0)                                   AS 平均播放量,
    ROUND(AVG(m.log_view), 3)                               AS 平均播放量对数,
    ROUND(AVG(m.like_rate) * 100, 2)                        AS 平均点赞率_pct,
    ROUND(AVG(m.coin_rate) * 100, 2)                        AS 平均投币率_pct,
    ROUND(AVG(m.favorite_rate) * 100, 2)                    AS 平均收藏率_pct,
    ROUND(AVG(m.share_rate) * 100, 2)                       AS 平均分享率_pct,
    ROUND(AVG(t.title_has_number) * 100, 1)                 AS 含数字标题占比_pct
FROM video_base b
JOIN video_metrics m ON b.bvid = m.bvid
JOIN text_features  t ON b.bvid = t.bvid
GROUP BY b.domain
ORDER BY 总播放量 DESC;

-- Q2 UP主粉丝层级（level: -1低 / 0中 / 1高）传播表现
SELECT
    CASE b.level WHEN 1 THEN '高层级' WHEN 0 THEN '中层级' ELSE '低层级' END AS 粉丝层级,
    COUNT(*)                                                                AS 视频数,
    COUNT(DISTINCT b.uid)                                                   AS UP主数,
    ROUND(AVG(b.fans), 0)                                                   AS 平均粉丝数,
    ROUND(AVG(b.view), 0)                                                   AS 平均播放量,
    ROUND(AVG(m.relative_view), 3)                                         AS 平均粉丝基准播放,
    ROUND(AVG(m.comment_rate) * 100, 2)                                    AS 平均评论率_pct
FROM video_base b
JOIN video_metrics m ON b.bvid = m.bvid
GROUP BY b.level
ORDER BY 平均粉丝数 DESC;

-- Q3 月度发布节奏与播放表现（时间序列口径，对应日报/周报）
SELECT
    substr(b.pubdate, 1, 7)                             AS 发布月份,
    COUNT(*)                                            AS 发布视频数,
    SUM(b.view)                                         AS 当月视频总播放,
    ROUND(AVG(b.view), 0)                               AS 单条平均播放,
    ROUND(AVG(m.like_rate) * 100, 2)                    AS 平均点赞率_pct
FROM video_base b
JOIN video_metrics m ON b.bvid = m.bvid
GROUP BY 发布月份
ORDER BY 发布月份;

-- Q4 UP主价值榜 TOP10（多表 JOIN + GROUP BY + 排序）
SELECT
    b.uid                                              AS uid,
    MIN(b.up_name)                                     AS UP主昵称,
    MIN(b.domain)                                      AS 知识领域,
    MIN(b.fans)                                        AS 粉丝量,
    COUNT(*)                                           AS 年发布视频数,
    SUM(b.view)                                        AS 全年总播放,
    ROUND(AVG(b.view), 0)                              AS 条均播放,
    ROUND(AVG(m.favorite_rate) * 100, 2)               AS 平均收藏率_pct
FROM video_base b
JOIN video_metrics m ON b.bvid = m.bvid
GROUP BY b.uid
ORDER BY 全年总播放 DESC
LIMIT 10;

-- Q5 内容互动漏斗（按领域）：播放 -> 点赞 -> 投币 -> 收藏 -> 分享，逐级转化率
SELECT
    b.domain                                              AS 知识领域,
    SUM(b.view)                                           AS 播放量,
    SUM(b.like)                                           AS 点赞数,
    SUM(b.coin)                                           AS 投币数,
    SUM(b.favorite)                                       AS 收藏数,
    SUM(b.share)                                          AS 分享数,
    ROUND(SUM(b.like)     * 1.0 / SUM(b.view) * 100, 2)   AS 点赞率_pct,
    ROUND(SUM(b.coin)     * 1.0 / SUM(b.like) * 100, 2)   AS 投币_点赞转化_pct,
    ROUND(SUM(b.favorite) * 1.0 / SUM(b.coin) * 100, 2)   AS 收藏_投币转化_pct,
    ROUND(SUM(b.share)    * 1.0 / SUM(b.favorite) * 100, 2) AS 分享_收藏转化_pct
FROM video_base b
GROUP BY b.domain
ORDER BY 播放量 DESC;
