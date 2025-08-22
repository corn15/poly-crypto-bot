from ts_convert import to_est_format
import time
import requests
import datetime
import threading
import schedule
import pytz
import json
pm_to_bn = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT"
}

class CryptoPredictionBot:
    def __init__(self):
        self.opening_prices = {}
        self.current_prices = {}
        self.clob_token_ids = {}
        self.token_prices = {}
        self.initialized = False

    def get_hour_start_timestamp(self):
        """获取当前整点的时间戳"""
        now = datetime.datetime.now(pytz.timezone('America/New_York'))
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        return int(hour_start.timestamp() * 1000)  # 转换为毫秒

    # def get_binance_opening_price(self, symbol, start_time):
    #     """获取Binance开盘价格"""
    #     try:
    #         url = f"https://api.binance.com/api/v3/klines"
    #         params = {
    #             'symbol': symbol,
    #             'interval': '1h',
    #             'startTime': start_time,
    #             'limit': 1
    #         }
    #         response = requests.get(url, params=params, timeout=10)
    #         response.raise_for_status()
    #         data = response.json()

    #         if data:
    #             opening_price = float(data[0][1])  # 开盘价在索引1
    #             print(f"✅ {symbol} 开盘价: {opening_price}")
    #             return opening_price
    #         return None
    #     except Exception as e:
    #         print(f"❌ 获取{symbol}开盘价失败: {e}")
    #         return None

    # def get_crypto_price(self, symbol, time):
    #     try:
    #         url = f"https://polymarket.com/api/crypto/crypto-price/"
    #         params = {
    #             'symbol': symbol,
    #             'eventStartTime': time,
    #             'variant': 'hourly',
    #             'endDate': time
    #         }

    #         response = requests.get(url, params=params, timeout=10)
    #         response.raise_for_status()
    #         data = response.json()
    #         if data:
    #             price = float(data.get('openPrice', 0))
    #             print(f"✅ {symbol} 加密貨幣價格: {price}")
    #             return price
    #         return None
    #     except Exception as e:
    #         print(f"❌ 獲取{symbol}加密貨幣價格失敗: {e}")
    #         return None

    def get_polymarket_tokens(self, crypto_name):
        """获取Polymarket市场的token信息"""
        try:
            # 获取当前时间的EST格式
            current_timestamp = int(time.time())
            time_format = to_est_format(current_timestamp)

            # 构建URL
            slug = f"{crypto_name}-up-or-down-{time_format}"
            url = f"https://gamma-api.polymarket.com/markets"
            params = {'slug': slug}

            print(f"🔍 搜尋 Polymarket 市場: {slug}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and len(data) > 0:
                clob_token_ids = json.loads(data[0].get('clobTokenIds', []))
                market_title = data[0].get('question', 'Unknown')
                print(f"✅ 找到市場: {market_title}")
                print(clob_token_ids)
                if len(clob_token_ids) >= 2:
                    print(clob_token_ids[0])
                    print(clob_token_ids[1])
                    up_token = clob_token_ids[0]
                    down_token = clob_token_ids[1]
                    print(f" up_token: {up_token}")
                    print(f" down_token: {down_token}")
                    return {'up': up_token, 'down': down_token}
                else:
                    print("⚠️  市場沒有足夠的 token")
            else:
                print(f"❌ 沒有找到相應的市場: {slug}")
            return None
        except Exception as e:
            print(f"❌ 获取{crypto_name} Polymarket tokens失败: {e}")
            return None

    def get_binance_current_price(self, symbol):
        """获取Binance当前价格"""
        try:
            url = f"https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': symbol}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            current_price = float(data['price'])
            return current_price
        except Exception as e:
            print(f"❌ 获取{symbol}当前价格失败: {e}")
            return None

    def get_polymarket_token_price(self, token_id):
        """获取Polymarket token价格"""
        try:
            url = f"https://clob.polymarket.com/price"
            params = {
                'token_id': token_id,
                'side': 'buy'
            }
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 404:
                # Token 沒有訂單簿是常見的情況
                return None

            response.raise_for_status()
            data = response.json()

            price = float(data.get('price', 0))
            return price
        except Exception as e:
            print(f"⚠️  Token {token_id[:20]}... 價格獲取失敗: {e}")
            return None

    def initialize_data(self):
        """初始化数据 - 在整点后30秒执行"""
        print("\n" + "="*50)
        print("🚀 初始化數據")
        print("="*50)
        hour_start = self.get_hour_start_timestamp()

        # 获取开盘价格
        for crypto_name, symbol in pm_to_bn.items():
            print(f"\n📊 處理 {crypto_name.upper()}...")

            opening_price = self.get_binance_opening_price(symbol, hour_start)
            if opening_price:
                self.opening_prices[crypto_name] = opening_price

            # 获取Polymarket tokens
            tokens = self.get_polymarket_tokens(crypto_name)
            if tokens:
                self.clob_token_ids[crypto_name] = tokens

        self.initialized = True
        print("\n✅ 初始化完成！")

    def update_prices(self):
        """更新价格 - 每分钟执行"""
        if not self.initialized:
            print("⚠️  尚未初始化，跳過價格更新")
            return

        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n📈 [{current_time}] 更新價格...")

        # 更新Binance价格
        for crypto_name, symbol in pm_to_bn.items():
            current_price = self.get_binance_current_price(symbol)
            if current_price:
                self.current_prices[crypto_name] = current_price

        # 更新Polymarket token价格
        for crypto_name, tokens in self.clob_token_ids.items():
            if crypto_name not in self.token_prices:
                self.token_prices[crypto_name] = {}
            for direction, token_id in tokens.items():
                price = self.get_polymarket_token_price(token_id)
                if price:
                    self.token_prices[crypto_name][direction] = price

        # 显示当前状态
        self.display_status()

    def display_status(self):
        """显示当前状态"""
        print("\n" + "="*30 + " 當前狀態 " + "="*30)

        for crypto_name in pm_to_bn.keys():
            print(f"\n🪙 {crypto_name.upper()}:")

            if crypto_name in self.opening_prices:
                opening = self.opening_prices[crypto_name]
                print(f"  🏁 開盤價: ${opening:,.2f}")

            if crypto_name in self.current_prices:
                current = self.current_prices[crypto_name]
                opening = self.opening_prices.get(crypto_name, 0)

                if opening > 0:
                    change = ((current - opening) / opening) * 100
                    if change > 0:
                        direction_emoji = "📈"
                        direction_text = "上漲"
                        color = "🟢"
                    elif change < 0:
                        direction_emoji = "📉"
                        direction_text = "下跌"
                        color = "🔴"
                    else:
                        direction_emoji = "➡️"
                        direction_text = "持平"
                        color = "🟡"
                    print(f"  💰 當前價: ${current:,.2f} {direction_emoji} {color} {direction_text} {abs(change):.2f}%")
                else:
                    print(f"  💰 當前價: ${current:,.2f}")

            if crypto_name in self.token_prices:
                tokens = self.token_prices[crypto_name]
                print("  🎯 Polymarket 價格:")
                if 'up' in tokens:
                    print(f"     📈 UP Token: ${tokens['up']:.3f}")
                if 'down' in tokens:
                    print(f"     📉 DOWN Token: ${tokens['down']:.3f}")
                # 計算隱含機率
                if 'up' in tokens and 'down' in tokens:
                    up_price = tokens['up']
                    down_price = tokens['down']
                    total = up_price + down_price
                    if total > 0:
                        up_prob = (up_price / total) * 100
                        down_prob = (down_price / total) * 100
                        print(f"     🎲 隱含機率: UP {up_prob:.1f}% | DOWN {down_prob:.1f}%")
            else:
                print("  ⚠️  Polymarket 價格暫不可用")

        print("="*70)

    def wait_for_next_initialization(self):
        """等待到下一個整點後30秒，或立即執行如果已經過了30秒"""
        now = datetime.datetime.now()

        # 檢查是否已經過了當前整點的30秒
        if now.minute > 0 or (now.minute == 0 and now.second >= 30):
            # 如果已經過了30秒，立即執行初始化
            print(f"⚡ 當前時間 {now.strftime('%H:%M:%S')} 已過整點30秒，立即執行初始化...")
            self.initialize_data()
            return

        # 否則等待到當前小時的30秒
        while True:
            now = datetime.datetime.now()
            target_time = now.replace(minute=0, second=30, microsecond=0)

            wait_seconds = (target_time - now).total_seconds()

            if wait_seconds <= 0:
                self.initialize_data()
                break
            else:
                print(f"⏰ 等待 {wait_seconds:.0f} 秒到整點後30秒進行初始化...")
                time.sleep(min(wait_seconds, 60))  # 最多等待60秒，然後重新檢查

    def start(self):
        """启动机器人"""
        print("🤖 啟動加密貨幣預測機器人...")
        print("📋 監控: BTC/USDT, ETH/USDT")
        print("🕐 初始化: 每小時整點後30秒")
        print("📊 價格更新: 每分鐘")
        print("-" * 50)

        # 设置定时任务 - 每分鐘更新價格
        schedule.every().minute.at(":00").do(self.update_prices)

        # 在單獨線程中等待初始化
        def wait_and_init():
            self.wait_for_next_initialization()

        init_thread = threading.Thread(target=wait_and_init)
        init_thread.daemon = True
        init_thread.start()

        # 主循環
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 機器人已停止")

def main():
    bot = CryptoPredictionBot()
    bot.get_crypto_price("BTC", "2025-08-20T00:00:00Z")
    # bot.start()

if __name__ == "__main__":
    main()
