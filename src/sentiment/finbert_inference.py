from transformers import pipeline


def load_finbert():

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )

    return sentiment_pipeline


def predict_sentiment(text, model):

    result = model(text)[0]

    return result["label"], result["score"]
