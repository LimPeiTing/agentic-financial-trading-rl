import numpy as np


def compute_metrics(portfolio_values):

    portfolio_values = np.array(portfolio_values)

    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1

    annual_return = np.mean(returns) * 252
    annual_volatility = np.std(returns) * np.sqrt(252)

    sharpe = (
        annual_return / (annual_volatility + 1e-8)
        if annual_volatility > 0
        else 0
    )

    peak = np.maximum.accumulate(portfolio_values)

    max_drawdown = np.max(
        (peak - portfolio_values) / (peak + 1e-8)
    )

    calmar = (
        annual_return / (max_drawdown + 1e-8)
        if max_drawdown > 0
        else 0
    )

    gains = returns[returns > 0]
    losses = returns[returns < 0]

    omega = (
        gains.sum() / abs(losses.sum())
        if len(losses) > 0
        else np.inf
    )

    downside = np.std(losses)

    sortino = (
        annual_return / (downside * np.sqrt(252) + 1e-8)
        if downside > 0
        else 0
    )

    p95 = np.percentile(returns, 95)
    p5 = abs(np.percentile(returns, 5))

    tail_ratio = p95 / (p5 + 1e-8)

    return {
        "Cumulative Return": cumulative_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
        "Omega Ratio": omega,
        "Sortino Ratio": sortino,
        "Tail Ratio": tail_ratio,
    }
