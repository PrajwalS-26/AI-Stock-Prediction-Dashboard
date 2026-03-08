import yfinance as yf
import pandas as pd
import time


def load_stock_data(ticker, period="10y"):

    df = pd.DataFrame()

    # Try downloading up to 3 times
    for _ in range(3):
        try:
            df = yf.download(
                ticker.upper(),
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

    # Standard price column
    df["Price"] = df["Close"]

    df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Price"]]

    df.dropna(inplace=True)

    return df




