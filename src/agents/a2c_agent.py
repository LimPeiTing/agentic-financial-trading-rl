from stable_baselines3 import A2C


def build_a2c_agent(env, config=None):

    if config is None:
        config = {
            "learning_rate": 3e-5,
            "gamma": 0.995,
            "ent_coef": 0.005,
            "net_arch": [256, 256],
            "seed": 42
        }

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        ent_coef=config["ent_coef"],
        policy_kwargs={"net_arch": config["net_arch"]},
        seed=config["seed"],
        verbose=1
    )

    return model
