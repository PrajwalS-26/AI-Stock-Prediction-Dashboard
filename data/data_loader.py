import yfinance as yf
import pandas as pd


def load_stock_data(ticker, period="10y"):

    df = yf.download(
        ticker,
        period=period,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df.reset_index(inplace=True)

    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    else:
        df["Price"] = df["Close"]

    df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Price"]]

    df.dropna(inplace=True)

    return dfhead())


