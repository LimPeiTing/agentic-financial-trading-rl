def compute_fundamental_ratios(df):

    df["Net_Profit_Margin"] = (
        df["netIncome"] / df["totalRevenue"]
    )

    df["Debt_to_Equity"] = (
        df["totalLiabilities"] / df["totalShareholderEquity"]
    )

    df["ROE"] = (
        df["netIncome"] / df["totalShareholderEquity"]
    )

    df["ROA"] = (
        df["netIncome"] / df["totalAssets"]
    )

    return df
