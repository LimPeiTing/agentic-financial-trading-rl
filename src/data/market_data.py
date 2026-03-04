import yfinance as yf
import pandas as pd


def download_market_data(tickers, start, end):

    data = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker"
    )

    dfs = []

    for ticker in tickers:

        df = data[ticker].copy()

        df["Ticker"] = ticker

        df.reset_index(inplace=True)

        dfs.append(df)

    market_df = pd.concat(dfs)

    return market_df
