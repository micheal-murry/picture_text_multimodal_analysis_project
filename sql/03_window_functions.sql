-- =====================================================================
-- 03 窗口函数分析
-- 用 ROW_NUMBER / RANK / LAG / NTILE / 累计分布等窗口函数做排名、趋势与帕累托分析。
-- =====================================================================

-- Q1 各知识领域播放量 TOP3 视频：RANK() 分区排名
WITH ranked AS (
    SELECT
        b.domain,
        b.bvid,
        b.title,
        b.up_name,
        b.view,
        RANK() OVER (PARTITION BY b.domain ORDER BY b.view DESC) AS 领域内排名
    FROM video_base b
)
SELECT * FROM ranked
WHERE 领域内排名 <= 3
ORDER BY domain, 领域内排名;

-- Q2 每个 UP 主内部播放量 TOP3 视频：ROW_NUMBER() + 子查询
SELECT * FROM (
    SELECT
        b.uid,
        b.up_name,
        b.title,
        b.view,
        ROW_NUMBER() OVER (PARTITION BY b.uid ORDER BY b.view DESC) AS UP内排名
    FROM video_base b
)
WHERE UP内排名 <= 3
ORDER BY uid, UP内排名
LIMIT 30;

-- Q3 播放量帕累托集中度：累计播放占比（头部内容的流量集中程度）
WITH ordered AS (
    SELECT
        bvid,
        view,
        ROW_NUMBER() OVER (ORDER BY view DESC)            AS rn,
        SUM(view) OVER ()                                  AS total_views
    FROM video_base
),
cum AS (
    SELECT
        rn,
        bvid,
        view,
        SUM(view) OVER (ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_views,
        total_views
    FROM ordered
)
SELECT
    COUNT(*)                                          AS 视频数,
    ROUND(MAX(CASE WHEN cum_views * 1.0 / total_views <= 0.50 THEN rn END) * 100.0 / COUNT(*), 1)
                                                      AS 贡献前50pct播放的视频占比_pct,
    ROUND(MAX(CASE WHEN cum_views * 1.0 / total_views <= 0.80 THEN rn END) * 100.0 / COUNT(*), 1)
                                                      AS 贡献前80pct播放的视频占比_pct
FROM cum;

-- Q4 单条视频 vs 同 UP 主平均水平：分区聚合窗口 + 相对倍数
SELECT
    b.up_name,
    b.title,
    b.view,
    ROUND(AVG(b.view) OVER (PARTITION BY b.uid), 0)                     AS 该UP平均播放,
    ROUND(b.view * 1.0 / AVG(b.view) OVER (PARTITION BY b.uid), 2)      AS 相对倍数,
    NTILE(4) OVER (ORDER BY b.fans)                                     AS 粉丝四分位
FROM video_base b
ORDER BY 相对倍数 DESC
LIMIT 15;

-- Q5 粉丝规模四分位组的传播表现对比：NTILE + 子查询 + 分组聚合
SELECT
    粉丝四分位,
    MIN(fans)                              AS 组内最少粉丝,
    MAX(fans)                              AS 组内最多粉丝,
    COUNT(*)                               AS 视频数,
    ROUND(AVG(view), 0)                    AS 平均播放,
    ROUND(AVG(relative_view), 3)           AS 平均粉丝基准播放
FROM (
    SELECT b.fans AS fans, b.view AS view, m.relative_view AS relative_view,
           NTILE(4) OVER (ORDER BY b.fans) AS 粉丝四分位
    FROM video_base b
    JOIN video_metrics m ON b.bvid = m.bvid
)
GROUP BY 粉丝四分位
ORDER BY 粉丝四分位;

-- Q6 月度发布量环比：LAG() 窗口函数（指标波动监控示例）
SELECT
    发布月份,
    发布视频数,
    LAG(发布视频数) OVER (ORDER BY 发布月份)                                  AS 上月发布数,
    ROUND((发布视频数 - LAG(发布视频数) OVER (ORDER BY 发布月份)) * 100.0
          / LAG(发布视频数) OVER (ORDER BY 发布月份), 1)                      AS 环比增速_pct
FROM (
    SELECT substr(pubdate, 1, 7) AS 发布月份, COUNT(*) AS 发布视频数
    FROM video_base
    GROUP BY 发布月份
)
ORDER BY 发布月份;
