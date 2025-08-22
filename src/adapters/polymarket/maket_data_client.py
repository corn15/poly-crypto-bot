
import httpx

class MarketDataClient:
    def __init__(self, api_key):
        self.base_url = "https://api.poly.market"

    def get_crypto_price(self, symbol, time):
        url = f"{self.base_url}/crypto/crypto-price/"
        params = {
            'symbol': symbol,
            'eventStartTime': time,
            'variant': 'hourly',
            'endDate': time
        }

        try:
            async with httpx.Client() as client:
                response = await client.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data:
                    price = float(data.get('openPrice', 0))
                    print(f"✅ {symbol} crypto price: {price}")
                    return price
                return None
        except Exception as e:
            print(f"❌ Failed to get {symbol} crypto price: {e}")
            return None
