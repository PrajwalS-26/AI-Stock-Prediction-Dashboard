import yfinance as yf
import pandas as pd
import time
import requests

def load_stock_data(ticker, period="10y"):
    ticker = ticker.upper()
    df = pd.DataFrame()

    # Feature to bypass Yahoo's bot detection on Streamlit Cloud
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    # Your existing retry logic
    for _ in range(5):
        try:
            df = yf.download(
                ticker,
                period=period,
                session=session,  # This clears the JSONDecodeError
                progress=False,
                threads=False,
                auto_adjust=True
            )

            if not df.empty:
                break

        except Exception:
            time.sleep(2)

    if df.empty:
        return pd.DataFrame()

    # Fix for MultiIndex columns often found in 2026 yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    # Your existing Close/Price logic
    if "Close" not in df.columns:
        raise ValueError("Yahoo returned unexpected data format")

    df["Price"] = df["Close"]

    # Your existing column selection
    df = df[["Date","Open","High","Low","Close","Volume","Price"]]
    
    # Your existing dropna
    df.dropna(inplace=True)

    return df






