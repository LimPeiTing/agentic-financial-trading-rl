import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.rmsf_env import RMSFEnv
from src.evaluation.metrics import compute_metrics


def run_backtest(model, test_df):

    state_features = [
        col for col in test_df.columns
        if col not in ["Date", "Ticker", "reward"]
    ]

    env = DummyVecEnv([
        lambda: RMSFEnv(
            df=test_df,
            state_features=state_features
        )
    ])

    obs = env.reset()

    portfolio_values = []

    while True:

        action, _ = model.predict(obs)

        obs, reward, done, info = env.step(action)

        portfolio_values.append(
            info[0]["portfolio_value"]
        )

        if done[0]:
            break

    metrics = compute_metrics(portfolio_values)

    return metrics, portfolio_values
