# Ensemble Deep Reinforcement Learning for Stock Trading

## Project Overview
This project implements an ensemble deep reinforcement learning (DRL) strategy for stock trading. It utilizes multiple DRL algorithms to optimize portfolio performance and compares the results against the Dow Jones Industrial Average benchmark.

## Key Features
- Ensemble strategy using A2C, PPO, DDPG, SAC, and TD3 algorithms
- Custom OpenAI Gym trading environment
- Feature engineering with technical indicators (MACD, RSI, CCI, DX)
- Backtesting framework with transaction costs and dynamic rebalancing
- Performance visualization and analysis

## Requirements
- Python 3.7+
- FinRL
- Libraries: pandas, numpy, matplotlib, gym, yfinance, plotly

Results
The ensemble strategy demonstrated superior risk-adjusted returns compared to the Dow Jones Industrial Average over the backtesting period.

Future Work
- Incorporate more advanced feature engineering techniques
- Experiment with different DRL algorithms and hyperparameters
- Expand to a larger universe of stocks and asset classes
