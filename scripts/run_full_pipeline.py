import pandas as pd

# Data
from src.data.market_data import download_market_data
from src.data.data_integration import merge_datasets

# Features
from src.features.technical_indicators import compute_technical_indicators
from src.features.fundamental_ratios import compute_fundamental_ratios
from src.features.dataset_builder import build_rl_dataset

# Sentiment
from src.sentiment.finbert_inference import load_finbert, predict_sentiment
from src.sentiment.sentiment_features import aggregate_daily_sentiment

# Training
from src.training.train_agents import train_agent

# Evaluation
from src.evaluation.ablation_study import run_ablation


# Config

TICKERS = ["AAPL", "MSFT", "GOOGL"]
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"

TRAIN_PATH = "data/rl_ready/train_state_ready.csv"
TEST_PATH = "data/rl_ready/test_state_ready.csv"



# Step 1 — Data Pipeline

def run_data_pipeline():

    print("Downloading market data...")

    market_df = download_market_data(
        TICKERS,
        START_DATE,
        END_DATE
    )

    return market_df



# Step 2 — Feature Engineering

def run_feature_engineering(df):

    print("Computing technical indicators...")

    df = compute_technical_indicators(df)

    print("Computing fundamental ratios...")

    df = compute_fundamental_ratios(df)

    return df



# Step 3 — Sentiment Pipeline

def run_sentiment_pipeline(df):

    print("Running FinBERT sentiment...")

    finbert = load_finbert()

    # example placeholder
    df["label"] = "neutral"
    df["score"] = 0.5

    sentiment_df = aggregate_daily_sentiment(df)

    return sentiment_df



# Step 4 — Dataset Construction

def build_dataset(df):

    print("Building RL dataset...")

    train_df, test_df = build_rl_dataset(df)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    return train_df, test_df



# Step 5 — Train RL Agents

def train_agents():

    print("Training PPO...")
    train_agent("ppo", TRAIN_PATH)

    print("Training A2C...")
    train_agent("a2c", TRAIN_PATH)

    print("Training DDPG...")
    train_agent("ddpg", TRAIN_PATH)


# Step 6 — Evaluation

def run_evaluation():

    print("Running ablation study...")

    test_df = pd.read_csv(TEST_PATH)

    from stable_baselines3 import PPO

    model = PPO.load("models/ppo_model.zip")

    results = run_ablation(model, test_df)

    results.to_csv("results/ablation_results.csv", index=False)

    print(results)



# Main

def main():

    df = run_data_pipeline()

    df = run_feature_engineering(df)

    sentiment_df = run_sentiment_pipeline(df)

    df = merge_datasets(df, sentiment_df, sentiment_df)

    train_df, test_df = build_dataset(df)

    train_agents()

    run_evaluation()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
