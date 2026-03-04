from stable_baselines3 import DDPG


def build_ddpg_agent(env, config=None):

    if config is None:
        config = {
            "learning_rate": 1e-4,
            "net_arch": [256, 256],
            "seed": 42
        }

    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        policy_kwargs={"net_arch": config["net_arch"]},
        seed=config["seed"],
        verbose=1
    )

    return model
