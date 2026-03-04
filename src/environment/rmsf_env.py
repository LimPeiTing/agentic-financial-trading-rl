import gym
from gym import spaces
import numpy as np
import pandas as pd


class RMSFEnv(gym.Env):
    """
    RMS-F Environment
    -----------------
    Multi-asset reinforcement learning trading environment.

    Tracks full portfolio performance metrics:
    - Annual Return
    - Cumulative Return
    - Annual Volatility
    - Sharpe Ratio
    - Calmar Ratio
    - Max Drawdown
    - Omega Ratio
    - Sortino Ratio
    - Tail Ratio
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        state_features: list,
        price_col: str = "Adj Close",
        initial_cash: float = 1_000_000,
        max_trade_size: int = 100,
        transaction_cost_pct: float = 0.0005,
        lambda_penalty: float = 0.5,
        allow_short: bool = False,
        reward_scale: float = 100.0,
        normalize_state: bool = True,
    ):

        super().__init__()

        self.df = df.copy()

        if "Date" not in df.columns or "Ticker" not in df.columns:
            raise ValueError("DataFrame must contain 'Date' and 'Ticker' columns.")

        if price_col not in df.columns:
            raise ValueError(f"{price_col} not found in dataframe.")

        self.state_features = [f for f in state_features if f in df.columns]

        if len(self.state_features) == 0:
            raise ValueError("No valid state features found.")

        self.price_col = price_col

        self.tickers = sorted(df["Ticker"].unique())
        self.dates = sorted(df["Date"].unique())

        self.n_tickers = len(self.tickers)
        self.n_features = len(self.state_features)

        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.lambda_penalty = lambda_penalty
        self.reward_scale = reward_scale
        self.allow_short = allow_short
        self.max_trade_size = max_trade_size
        self.normalize_state = normalize_state

        # observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_tickers * self.n_features,),
            dtype=np.float32,
        )

        # action space
        self.action_space = spaces.Box(
            low=-max_trade_size,
            high=max_trade_size,
            shape=(self.n_tickers,),
            dtype=np.float32,
        )

        self.reset()

   
    # Reset Environment
    

    def reset(self):

        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_tickers)

        self.portfolio_values = [self.initial_cash]
        self.returns_history = []
        self.drawdowns = []

        return self._get_observation()

    
    # Observation

    def _get_observation(self):

        date = self.dates[self.current_step]

        subset = (
            self.df[self.df["Date"] == date]
            .set_index("Ticker")
            .reindex(self.tickers)
        )

        obs = subset[self.state_features].fillna(0).values.flatten()

        if self.normalize_state:
            mean = np.mean(obs)
            std = np.std(obs)
            obs = (obs - mean) / (std + 1e-8)

        return obs.astype(np.float32)


    # Step

    def step(self, action):

        action = np.clip(action, -self.max_trade_size, self.max_trade_size)

        prices_today = (
            self.df[self.df["Date"] == self.dates[self.current_step]]
            .set_index("Ticker")
            .reindex(self.tickers)[self.price_col]
            .fillna(method="ffill")
            .fillna(method="bfill")
            .values
        )

        prices_today = np.where(prices_today <= 0, 1e-8, prices_today)

        for i in range(self.n_tickers):

            trade_size = action[i]

            if not self.allow_short and trade_size < 0:
                trade_size = -min(abs(trade_size), self.holdings[i])

            trade_value = trade_size * prices_today[i]

            cost = abs(trade_value) * self.transaction_cost_pct

            if self.cash - (trade_value + cost) < -self.initial_cash * 0.2:
                continue

            self.cash -= trade_value + cost
            self.holdings[i] += trade_size

        self.current_step += 1

        done = self.current_step >= len(self.dates) - 1

        next_prices = (
            self.df[self.df["Date"] == self.dates[self.current_step]]
            .set_index("Ticker")
            .reindex(self.tickers)[self.price_col]
            .fillna(method="ffill")
            .fillna(method="bfill")
            .values
        )

        portfolio_value = self.cash + np.sum(self.holdings * next_prices)

        self.portfolio_values.append(portfolio_value)

        prev_value = self.portfolio_values[-2]
        curr_value = self.portfolio_values[-1]

        portfolio_return = (curr_value - prev_value) / prev_value

        drawdown = (max(self.portfolio_values) - curr_value) / max(
            self.portfolio_values
        )

        reward = (
            portfolio_return - self.lambda_penalty * drawdown
        ) * self.reward_scale

        self.returns_history.append(portfolio_return)
        self.drawdowns.append(drawdown)

        obs = self._get_observation()

        info = {
            "portfolio_value": float(curr_value),
            "return": float(portfolio_return),
            "drawdown": float(drawdown),
            "cash": float(self.cash),
            "holdings": self.holdings.copy(),
        }

        return obs, reward, done, info
