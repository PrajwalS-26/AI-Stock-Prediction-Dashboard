import yfinance as yf
import pandas as pd
import time
import requests

def load_stock_data(ticker, period="10y"):
    ticker = ticker.upper()
    df = pd.DataFrame()

    # 1. FIX: Use a session to bypass Yahoo's bot detection on Streamlit
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    for _ in range(5):
        try:
            df = yf.download(
                ticker,
                period=period,
                session=session,  # Critical for Streamlit Cloud
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

    # 2. FIX: Handle the MultiIndex columns (The "No Data Found" cause)
    # This flattens ('Close', 'AAPL') back to just 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    # Ensure Close exists after flattening
    if "Close" not in df.columns:
        raise ValueError(f"No Close price found for {ticker}")

    # Create Price column (Your original logic)
    df["Price"] = df["Close"]

    # Keep needed columns
    df = df[["Date","Open","High","Low","Close","Volume","Price"]]

    return df







