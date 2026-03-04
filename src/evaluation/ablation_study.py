import pandas as pd

from src.evaluation.backtest import run_backtest


FEATURE_GROUPS = {

    "T": ["technical"],

    "F": ["fundamental"],

    "S": ["sentiment"],

    "TF": ["technical", "fundamental"],

    "TS": ["technical", "sentiment"],

    "FS": ["fundamental", "sentiment"],

    "TFS": ["technical", "fundamental", "sentiment"]
}


def run_ablation(model, test_df):

    results = []

    for group in FEATURE_GROUPS:

        print(f"Running ablation: {group}")

        metrics, _ = run_backtest(model, test_df)

        metrics["Feature Set"] = group

        results.append(metrics)

    results_df = pd.DataFrame(results)

    return results_df
