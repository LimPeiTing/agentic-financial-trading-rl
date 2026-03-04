import pandas as pd


def build_rl_dataset(df):

    df = df.sort_values(["Date", "Ticker"])

    df.fillna(method="ffill", inplace=True)

    train_df = df[df["Date"] < "2022-01-01"]

    test_df = df[df["Date"] >= "2022-01-01"]

    return train_df, test_df
