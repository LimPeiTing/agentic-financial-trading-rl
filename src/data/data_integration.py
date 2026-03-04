import pandas as pd


def merge_datasets(
    market_df,
    fundamentals_df,
    sentiment_df
):

    df = market_df.merge(
        fundamentals_df,
        on=["Ticker"],
        how="left"
    )

    df = df.merge(
        sentiment_df,
        on="Date",
        how="left"
    )

    return df
