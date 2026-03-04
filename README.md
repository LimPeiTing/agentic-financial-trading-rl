## Agentic Financial Trading with Reinforcement Learning and Sentiment Analysis

A research project that explores how financial sentiment analysis and reinforcement learning (RL) can be combined to support automated trading decisions under controlled offline backtesting conditions.

This project was developed as part of a Master’s thesis in Data Science and focuses on designing a modular AI trading framework that integrates:

- financial market data

- fundamental indicators

- sentiment signals

- reinforcement learning agents

- an optional LLM-based policy oversight layer

The system evaluates trading strategies using offline historical data only and emphasizes risk-aware portfolio metrics rather than profit claims.

---

## Project Overview

The goal of this project is to study how different sources of financial information can improve trading decisions when used as inputs to reinforcement learning agents.

The system combines three types of financial signals:

- Market data

- Fundamental company indicators

- Financial sentiment signals

These signals form the state representation for reinforcement learning agents, which learn trading strategies through a custom portfolio simulation environment.

---

## Project Highlights

• Designed an end-to-end **AI trading research pipeline** integrating financial data, sentiment analysis, and reinforcement learning.

• Developed a **custom multi-asset reinforcement learning trading environment** using OpenAI Gym to simulate portfolio allocation and transaction costs.

• Integrated **FinBERT financial sentiment analysis** to convert textual information into quantitative signals for reinforcement learning agents.

• Implemented and evaluated multiple reinforcement learning algorithms (**PPO, A2C, DDPG**) for portfolio decision making.

• Conducted **feature ablation studies** to analyze the impact of technical, fundamental, and sentiment features on trading performance.

• Proposed an **optional LLM policy oversight layer** to review agent decisions and explore agentic AI governance concepts.

• Built a **fully modular and reproducible research pipeline** covering data collection, feature engineering, model training, evaluation, and experimentation.

---

## System Architecture

```mermaid
flowchart TD

A[Market Data<br>Yahoo Finance]
B[Fundamental Data<br>Alpha Vantage]
C[Text Data<br>News / Reddit]

A --> D[Data Collection]
B --> D
C --> E[FinBERT Sentiment]

D --> F[Feature Engineering]
E --> G[Sentiment Index]

F --> H[State Representation]
G --> H

H --> I[RL Trading Environment]

I --> J[PPO]
I --> K[A2C]
I --> L[DDPG]

J --> M[Backtesting]
K --> M
L --> M
M --> N[Ablation Study]
N --> O[Optional LLM Policy Annotator]

```

---

## System Workflow

The full pipeline executes in the following stages:
1. Collect financial market data
2. Engineer financial features
3. Generate sentiment signals using FinBERT
4. Build RL-ready datasets
5. Train reinforcement learning agents
6. Evaluate strategies using offline backtesting
7. Perform ablation studies
8. Optionally apply LLM policy moderation

---

## Data Sources

The project integrates multiple financial data sources.

- Market Data

- Historical stock market data including:

- Open

- High

- Low

- Close

- Adjusted Close

- Volume

Collected using the Yahoo Finance API.

---

## Fundamental Indicators

Company financial metrics used to capture business fundamentals.

Examples include:

- earnings per share

- revenue

- net income

- financial ratios

Collected using the Alpha Vantage API.

---
## Sentiment Signals

Financial sentiment signals are extracted from textual sources.

The project uses FinBERT, a transformer model trained specifically for financial language.

Sentiment outputs are converted into numerical signals and aggregated into a daily sentiment index.

---

## Feature Engineering

The system generates multiple feature groups used by the reinforcement learning agents.

Technical Indicators

Examples:

- RSI

- MACD

- CCI

- ADX

- price percentage changes

- trading volume

These indicators capture market momentum and price trends.

---

## Fundamental Ratios

Examples:

- Return on Equity (ROE)

- Return on Assets (ROA)

- Net Profit Margin

- Debt to Equity ratio

- Free Cash Flow

These represent company financial health.

---

## Sentiment Features

Examples:

- FinBERT sentiment scores

- aggregated daily sentiment index

- external sentiment indicators

These represent market perception and investor mood.

---

## Reinforcement Learning Environment

A custom OpenAI Gym environment was developed to simulate a multi-asset trading system.

Key characteristics:

- multi-asset portfolio

- continuous action space

- transaction costs

- portfolio value tracking

- risk-aware reward function

The environment receives actions from RL agents and updates portfolio positions accordingly.


---


## Reward Design

The reward function balances profitability and risk control.
Reward = portfolio_return − (drawdown_penalty × drawdown)
This discourages strategies that generate profits by taking excessive risk.

---

## Reinforcement Learning Agents

Three RL algorithms were evaluated:

PPO — Proximal Policy Optimization

A stable policy gradient method commonly used in financial RL research.

A2C — Advantage Actor Critic

An actor-critic algorithm that learns both policy and value functions.

DDPG — Deep Deterministic Policy Gradient

A continuous control algorithm suitable for portfolio allocation problems.

---

## Training Process

Agents are trained using historical market data under an offline simulation environment.

Training Dataset
      │
      ▼
RL Environment Simulation
      │
      ▼
Agent learns trading policy
      │
      ▼
Trained model saved

No live trading or real capital is involved.

---

## Trading Decision Process

This explains how a single trading decision is made.

```mermaid
flowchart LR

A[Market State<br>Technical + Fundamental + Sentiment]

A --> B[RL Agent]

B --> C[Proposed Trade Action]

C --> D{LLM Policy Annotator<br>Optional}

D -->|Approve| E[Execute Trade]

D -->|Reduce| F[Adjust Position Size]

D -->|Reject| G[Cancel Trade]

F --> E

E --> H[Portfolio Update]

H --> I[Reward Calculation]
```

---

## Evaluation and Backtesting

All models are evaluated using out-of-sample historical data.

Performance is measured using standard financial metrics.

Examples include:

- Annual Return

- Cumulative Return

- Annual Volatility

- Sharpe Ratio

- Sortino Ratio

- Calmar Ratio

- Max Drawdown

- Omega Ratio

- Tail Ratio

These metrics provide a balanced evaluation of performance and risk.

---


## Ablation Study

An ablation study is conducted to evaluate how different feature groups affect agent performance.

Feature groups tested include:
Technical indicators only
Fundamental indicators only
Sentiment features only
Technical + Fundamental
Technical + Sentiment
Fundamental + Sentiment
Technical + Fundamental + Sentiment

This helps understand which types of financial signals contribute most to decision quality.

---

## Optional LLM Policy Annotator

The system also explores a concept called Agentic AI oversight.

An optional module can review trading actions before execution.
RL Agent proposes action
        │
        ▼
LLM Policy Annotator
        │
        ▼
Approve / Reduce / Reject trade

This module is not used during training and serves only as an experimental governance layer.

---

## Repository Structure
```bash

agentic-financial-trading-rl

data/
    raw/
    processed/
    rl_ready/

models/
results/

src/
    data/
    features/
    sentiment/
    environment/
    agents/
    training/
    evaluation/
    annotation/
    utils/

scripts/
    run_full_pipeline.py
    train_all_agents.py
    run_ablation.py

notebooks/
    thesis_experiments/
```
---

## Running the Full Pipeline

The entire system can be executed with a single command.

python scripts/run_full_pipeline.py

This will:

- collect and prepare data

- generate features

- compute sentiment signals

- train reinforcement learning agents

- evaluate strategies

- produce experiment results

---

## Key Contributions

This project demonstrates how multiple AI techniques can be combined within a financial decision framework.

Main contributions include:

- integration of financial sentiment analysis and reinforcement learning

- a custom multi-asset trading environment

- evaluation of multiple RL algorithms

- feature ablation analysis

- modular research pipeline design

- exploration of LLM-based policy moderation

---

## Important Notes

This project is a research study and not a trading recommendation system.

Key limitations:

- evaluation is performed using historical data only

- results are dependent on market conditions in the dataset

- the system is designed for research and experimentation

- No claims are made about real-world profitability.


---

