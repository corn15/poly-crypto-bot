# 加密貨幣 Polymarket 預測機器人

一個自動監控 Bitcoin 和 Ethereum 價格，並跟蹤 Polymarket 相關預測市場的機器人。

## 功能特點

### 🔄 自動執行時間表
- **智能初始化**:
  - 如果啟動時已過整點30秒，立即獲取當前整點的開盤價格
  - 否則等待到下一個整點後30秒進行初始化
- **價格更新**: 每分鐘獲取最新價格數據

### 📊 數據源
1. **Binance API**
   - 整點開盤價格 (K線數據)
   - 每分鐘當前價格更新
   - 支援 BTC/USDT 和 ETH/USDT

2. **Polymarket API**
   - 市場信息 (clobTokenIds)
   - UP/DOWN token 價格
   - 隱含市場機率計算

### 📈 監控內容
- **價格追蹤**: 開盤價 vs 當前價
- **漲跌幅**: 實時計算百分比變化
- **市場情緒**: Polymarket UP/DOWN token 價格
- **隱含機率**: 基於 token 價格計算的市場預期

## 安裝與運行

### 1. 安裝依賴
```bash
uv sync
```

### 2. 運行機器人
```bash
uv run python main.py
```

機器人將：
- 🟢 立即啟動並顯示當前狀態
- ⚡ 如果已過整點30秒，立即獲取當前整點開盤價並初始化
- ⏰ 否則等待到下一個整點後30秒進行完整初始化
- 📊 每分鐘更新價格數據

### 3. 停止機器人
按 `Ctrl+C` 停止機器人

## 輸出示例

```
🤖 啟動加密貨幣預測機器人...
📋 監控: BTC/USDT, ETH/USDT
🕐 初始化: 每小時整點後30秒
📊 價格更新: 每分鐘

============================== 當前狀態 ==============================

🪙 BITCOIN:
  🏁 開盤價: $122,954.03
  💰 當前價: $123,549.77 📈 🟢 上漲 0.48%
  🎯 Polymarket 價格:
     📈 UP Token: $0.520
     📉 DOWN Token: $0.480
     🎲 隱含機率: UP 52.0% | DOWN 48.0%

🪙 ETHEREUM:
  🏁 開盤價: $4,734.31
  💰 當前價: $4,759.95 📈 🟢 上漲 0.54%
  🎯 Polymarket 價格:
     📈 UP Token: $0.465
     📉 DOWN Token: $0.535
     🎲 隱含機率: UP 46.5% | DOWN 53.5%
======================================================================
```

## 技術細節

### API 端點
- **Binance K線**: `https://api.binance.com/api/v3/klines`
- **Binance 價格**: `https://api.binance.com/api/v3/ticker/price`
- **Polymarket 市場**: `https://gamma-api.polymarket.com/markets`
- **Polymarket 價格**: `https://clob.polymarket.com/price`

### 時間格式
Polymarket slug 使用 EST 時間格式: `bitcoin-up-or-down-august-13-7pm-et`

### 錯誤處理
- 網絡請求超時 (10秒)
- API 錯誤自動重試
- 404 錯誤 (無訂單簿) 正常處理
- 優雅的錯誤信息顯示

## 依賴項
- `pytz`: 時區處理
- `requests`: HTTP 請求
- `schedule`: 定時任務

## 項目結構
```
├── main.py           # 主程序
├── ts_convert.py     # 時間轉換工具
├── pyproject.toml    # 項目配置
└── README.md         # 說明文件
```
