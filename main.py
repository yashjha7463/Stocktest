import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings("ignore")

from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return

from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

import os

# Create the 'results' directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

def run_backtest(ticker):
    # Configure the start and end dates for training and testing
    TRAIN_START_DATE = '2010-01-01'
    TRAIN_END_DATE = '2021-10-01'
    TEST_START_DATE = '2021-10-01'
    TEST_END_DATE = '2023-03-01'

    # Download the stock data
    df = YahooDownloader(start_date=TRAIN_START_DATE,
                         end_date=TEST_END_DATE,
                         ticker_list=[ticker]).fetch_data()

    # Preprocess the data
    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=INDICATORS,
                         use_turbulence=True,
                         user_defined_feature=False)

    processed = fe.preprocess_data(df)
    processed = processed.copy()
    processed = processed.fillna(0)
    processed = processed.replace(np.inf, 0)

    # Set up the trading environment
    stock_dimension = len(processed.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "print_verbosity": 5
    }

    # Run the ensemble strategy
    rebalance_window = 63
    validation_window = 63
    ensemble_agent = DRLEnsembleAgent(df=processed,
                                      train_period=(TRAIN_START_DATE, TRAIN_END_DATE),
                                      val_test_period=(TEST_START_DATE, TEST_END_DATE),
                                      rebalance_window=rebalance_window,
                                      validation_window=validation_window,
                                      **env_kwargs)

    A2C_model_kwargs = {
        'n_steps': 5,
        'ent_coef': 0.005,
        'learning_rate': 0.007
    }
    PPO_model_kwargs = {
    "ent_coef": 0.01,
    "n_steps": 2048,
    "learning_rate": 0.00025,
    "batch_size": 128
}

    DDPG_model_kwargs = {
    "buffer_size": 10000,
    "learning_rate": 0.0005,
    "batch_size": 64
}

    TD3_model_kwargs = {
    "buffer_size": 10000,
    "learning_starts": 100,
    "batch_size": 100,
}

    SAC_model_kwargs = {
    "buffer_size": 1000,
    "learning_starts": 1000,
    "batch_size": 128,
    "ent_coef": "auto",
}

    timesteps_dict = {
    "a2c": 1000,
    "ppo": 1000,
    "ddpg": 1000,
    "sac": 1000,
    "td3": 1000,
}

    df_summary = ensemble_agent.run_ensemble_strategy(A2C_model_kwargs,
                                                   PPO_model_kwargs,
                                                   DDPG_model_kwargs,
                                                   SAC_model_kwargs,
                                                   TD3_model_kwargs,
                                                   timesteps_dict)

    # Compute the backtest results
    unique_trade_date = processed[(processed.date > TEST_START_DATE) & (processed.date <= TEST_END_DATE)].date.unique()
    df_trade_date = pd.DataFrame({'datadate': unique_trade_date})
    df_account_value = pd.DataFrame()
    for i in range(rebalance_window + validation_window, len(unique_trade_date) + 1, rebalance_window):
        temp = pd.read_csv(f'results/account_value_trade_ensemble_{i}.csv')
        df_account_value = pd.concat([df_account_value, temp], ignore_index=True)
    df_account_value = df_account_value.join(df_trade_date[validation_window:].reset_index(drop=True))

    # Download the Dow Jones Index data
    dji_data = YahooDownloader(start_date=df_account_value.loc[0, 'date'],
                               end_date=df_account_value.loc[len(df_account_value) - 1, 'date'],
                               ticker_list=['^DJI']).fetch_data()

    # Normalize the DJI data
    dji_data['dji'] = dji_data['close'] / dji_data['close'][0] * env_kwargs["initial_amount"]

    # Merge the ensemble and DJI data
    df_result_ensemble = pd.DataFrame({'date': df_account_value['date'], 'ensemble': df_account_value['account_value']})
    df_result_ensemble = df_result_ensemble.set_index('date')
    result = pd.merge(df_result_ensemble, dji_data[['date', 'dji']], left_index=True, right_on='date', how='left')
    result = result.set_index('date')

    # Get the column names from the result DataFrame
    column_names = list(result.columns)

    # Assign the column names to the result DataFrame
    result.columns = column_names

    return result
