import os
import time
import random
import requests
import pandas as pd

# ==================== 配置区 ====================
INPUT_FILE = 'up_list_raw.xlsx'
OUTPUT_FILE = 'up_list_final.xlsx'
FAILED_FILE = 'api_failed_list.txt'

# 填入 SESSDATA 更稳，但不填也有很大概率能跑通
SESSDATA = "" 

EXCLUDE_CERT_KEYWORDS = ["机构", "公司", "媒体", "电视台", "政府", "学校"]

# ==================== 核心逻辑 ====================
def get_headers():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://www.bilibili.com/'
    }
    if SESSDATA:
        headers['Cookie'] = f'SESSDATA={SESSDATA}'
    return headers

class BiliApiSpider:
    def __init__(self):
        self.headers = get_headers()
        self.success_data = []
        self.processed_uids = set()
        self.request_count = 0
        self.load_checkpoints()

    def load_checkpoints(self):
        if os.path.exists(OUTPUT_FILE):
            try:
                df = pd.read_excel(OUTPUT_FILE)
                self.success_data = df.to_dict('records')
                self.processed_uids = set(df['UID'].astype(str).tolist())
                print(f"[系统] 成功加载断点，已跳过 {len(self.processed_uids)} 个已采集的 UP 主。")
            except Exception as e:
                pass

    def save_data(self):
        if not self.success_data:
            return
        df = pd.DataFrame(self.success_data, columns=['昵称', '粉丝量', 'UID', '认证信息'])
        df.to_excel(OUTPUT_FILE, index=False)

    def log_failed(self, uid, reason):
        with open(FAILED_FILE, 'a', encoding='utf-8') as f:
            f.write(f"UID: {uid}, Reason: {reason}, Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def handle_rate_limit(self):
        self.request_count += 1
        if self.request_count % 100 == 0:
            print("\n[策略] 已连续请求 100 次，触发安全熔断，强制暂停 5 分钟...")
            time.sleep(300)
        else:
            # 缩短了一点点休眠时间，因为现在只发 1 个请求了
            time.sleep(random.uniform(2.0, 3.5))

    def run(self):
        target_path = os.path.abspath(INPUT_FILE)
        if not os.path.exists(target_path):
            print(f"[错误] 找不到输入文件：{target_path}")
            return

        try:
            input_df = pd.read_excel(INPUT_FILE)
            uid_list = []
            if 'UID' in input_df.columns:
                uid_list = input_df['UID'].astype(str).tolist() 
            elif 'uid' in input_df.columns:
                uid_list = input_df['uid'].astype(str).tolist()
            else:
                return
        except Exception as e:
            return

        print(f"[系统] 成功读取 {len(uid_list)} 条 UID 数据，开始执行 API 批量查询...\n")

        for index, uid in enumerate(uid_list):
            uid = uid.strip()
            if not uid or uid in self.processed_uids:
                continue

            print(f"[{index + 1}/{len(uid_list)}] 查询 UID: {uid} ...", end=" ")
            self.handle_rate_limit()

            # --- 核心改动：使用单接口获取所有信息 ---
            card_url = f"https://api.bilibili.com/x/web-interface/card?mid={uid}"
            
            try:
                res = requests.get(card_url, headers=self.headers, timeout=10)
                
                if res.status_code == 429:
                    print("\n[警告] 触发限流 (429 Too Many Requests)，挂起 10 分钟...")
                    time.sleep(600)
                    res = requests.get(card_url, headers=self.headers, timeout=10)
                    
                data = res.json()
            except Exception as e:
                print(f"失败 (网络异常: {str(e)})")
                self.log_failed(uid, str(e))
                continue

            if data.get('code') != 0:
                print(f"失败 (接口错误: {data.get('message', '未知代码 '+str(data.get('code')))})")
                self.log_failed(uid, f"API Code: {data.get('code')}")
                continue

            # 数据解析
            try:
                card_info = data['data']['card']
                nickname = card_info.get('name', '未知')
                fans_num = card_info.get('fans', 0)
                
                # B 站此接口的 Official 首字母是大写的
                cert_text = ""
                if 'Official' in card_info and 'title' in card_info['Official']:
                    cert_text = card_info['Official']['title']
            except KeyError:
                print("失败 (数据结构解析异常)")
                self.log_failed(uid, "Parse Error")
                continue

            # 过滤逻辑：粉丝量 < 1万
            if fans_num < 10000:
                print(f"跳过 (粉丝不足: {fans_num})")
                self.processed_uids.add(uid)
                continue

            # 过滤逻辑：非个人账号
            is_excluded = any(keyword in cert_text for keyword in EXCLUDE_CERT_KEYWORDS)
            if is_excluded:
                print(f"跳过 (认证排除: {cert_text})")
                self.processed_uids.add(uid)
                continue

            # 数据入库
            print(f"成功 -> {nickname} (粉丝: {fans_num}, 认证: {cert_text if cert_text else '无'})")
            self.success_data.append({
                '昵称': nickname,
                '粉丝量': fans_num,
                'UID': uid,
                '认证信息': cert_text
            })
            self.processed_uids.add(uid)

            if len(self.success_data) % 10 == 0:
                self.save_data()

        self.save_data()
        print(f"\n[系统] 所有任务执行完毕！最终数据已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    spider = BiliApiSpider()
    spider.run()