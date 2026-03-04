import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.rmsf_env import RMSFEnv
from src.agents.ppo_agent import build_ppo_agent
from src.agents.a2c_agent import build_a2c_agent
from src.agents.ddpg_agent import build_ddpg_agent


def create_env(data_path):

    df = pd.read_csv(data_path)

    state_features = [
        col for col in df.columns
        if col not in ["Date", "Ticker", "reward"]
    ]

    env = DummyVecEnv([
        lambda: RMSFEnv(df=df, state_features=state_features)
    ])

    return env


def train_agent(agent_name, train_path, timesteps=50000):

    env = create_env(train_path)

    if agent_name == "ppo":
        model = build_ppo_agent(env)

    elif agent_name == "a2c":
        model = build_a2c_agent(env)

    elif agent_name == "ddpg":
        model = build_ddpg_agent(env)

    else:
        raise ValueError("Unknown agent type")

    model.learn(total_timesteps=timesteps)

    model.save(f"models/{agent_name}_model")

    return model
