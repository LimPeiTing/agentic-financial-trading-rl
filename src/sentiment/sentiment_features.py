import pandas as pd


def aggregate_daily_sentiment(df):

    sentiment_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    df["sentiment_score"] = df["label"].map(sentiment_map)

    daily = df.groupby("Date")["sentiment_score"].mean()

    daily = daily.reset_index()

    daily.rename(
        columns={"sentiment_score": "daily_sentiment_index"},
        inplace=True
    )

    return daily
