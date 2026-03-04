import pandas as pd

from stable_baselines3 import PPO

from src.evaluation.ablation_study import run_ablation


TEST_DATA = "data/rl_ready/test_state_ready.csv"

MODEL_PATH = "models/ppo_model.zip"


def main():

    test_df = pd.read_csv(TEST_DATA)

    model = PPO.load(MODEL_PATH)

    results = run_ablation(model, test_df)

    print(results)

    results.to_csv(
        "results/ablation_results.csv",
        index=False
    )


if __name__ == "__main__":
    main()
