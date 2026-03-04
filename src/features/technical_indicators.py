import pandas as pd
import ta


def compute_technical_indicators(df):

    df["RSI"] = ta.momentum.rsi(df["Adj Close"])

    macd = ta.trend.macd(df["Adj Close"])

    df["MACD"] = macd

    df["CCI"] = ta.trend.cci(
        df["High"],
        df["Low"],
        df["Adj Close"]
    )

    df["ADX"] = ta.trend.adx(
        df["High"],
        df["Low"],
        df["Adj Close"]
    )

    return df
