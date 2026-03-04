from src.training.train_agents import train_agent


TRAIN_DATA = "data/rl_ready/train_state_ready.csv"


def main():

    print("Training PPO...")
    train_agent("ppo", TRAIN_DATA)

    print("Training A2C...")
    train_agent("a2c", TRAIN_DATA)

    print("Training DDPG...")
    train_agent("ddpg", TRAIN_DATA)


if __name__ == "__main__":
    main()
