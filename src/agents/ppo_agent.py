from stable_baselines3 import PPO


def build_ppo_agent(env, config=None):

    if config is None:
        config = {
            "learning_rate": 3e-5,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "net_arch": [256, 256],
            "seed": 42
        }

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        ent_coef=config["ent_coef"],
        clip_range=config["clip_range"],
        policy_kwargs={"net_arch": config["net_arch"]},
        seed=config["seed"],
        verbose=1
    )

    return model
