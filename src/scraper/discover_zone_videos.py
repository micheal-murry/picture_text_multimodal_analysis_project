import requests
import time
import random

def crawl_bilibili_knowledge_videos(pages=3, delay=2):
    """
    爬取 B站 知识区视频的 UID 和链接
    :param pages: 爬取页数（默认3页）
    :param delay: 请求间隔（秒，避免被封）
    """
    # 请求头（模拟浏览器，关键！）
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.bilibili.com/",
    }

    # 知识区分区 ID（tid=36 代表知识区，可根据需求调整）
    tid = 36
    video_info = []

    for page in range(1, pages + 1):
        try:
            # B站 分区视频 API（最新投稿）
            url = f"https://api.bilibili.com/x/web-interface/newlist?rid={tid}&pn={page}&ps=20"
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # 检查请求是否成功
            data = response.json()

            # 解析数据
            if data["code"] == 0:
                videos = data["data"]["archives"]
                for video in videos:
                    video_info.append({
                        "uid": video["owner"]["mid"],  # UP主 UID
                        "video_url": f"https://www.bilibili.com/video/{video['bvid']}",  # 视频链接
                        "title": video["title"],  # 视频标题（方便查看）
                    })
                print(f"第 {page} 页爬取成功，已获取 {len(video_info)} 条数据")
            else:
                print(f"API 返回错误：{data['message']}")
                break

        except Exception as e:
            print(f"第 {page} 页爬取失败：{str(e)}")
            break

        # 随机延迟，避免被反爬
        time.sleep(delay + random.uniform(0, 1))

    return video_info


if __name__ == "__main__":
    # 爬取 2 页数据（可自行调整）
    result = crawl_bilibili_knowledge_videos(pages=2)
    
    # 打印结果
    for item in result:
        print(f"UP主 UID: {item['uid']} | 视频链接: {item['video_url']} | 标题: {item['title']}")