import requests
import pandas as pd


def fetch_fundamentals(ticker, api_key):

    url = (
        f"https://www.alphavantage.co/query?"
        f"function=EARNINGS&symbol={ticker}&apikey={api_key}"
    )

    r = requests.get(url)

    data = r.json()

    earnings = pd.DataFrame(data["quarterlyEarnings"])

    earnings["Ticker"] = ticker

    return earnings
