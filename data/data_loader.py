import yfinance as yf
import pandas as pd
import time


def load_stock_data(ticker, period="10y"):

    ticker = ticker.upper()

    df = pd.DataFrame()

    for _ in range(5):
        try:
            df = yf.download(
                ticker,
                period=period,
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

    df.reset_index(inplace=True)

    # Ensure Close exists
    if "Close" not in df.columns:
        raise ValueError("Yahoo returned unexpected data format")

    # Create Price column
    df["Price"] = df["Close"]

    # Keep needed columns
    df = df[["Date","Open","High","Low","Close","Volume","Price"]]

    return df





