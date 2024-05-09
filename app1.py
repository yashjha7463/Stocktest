import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from main1 import run_backtest

def main():
    st.title("Ensemble Trading Strategy Backtest")

    # Get user input for the stock ticker
    ticker = st.text_input("Enter the stock ticker (e.g., DOW_30_TICKER):")
    Choices = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW']
    
    st.write("Here is the list of stock tickers:")
    st.write(Choices)


    # Run the backtest
    result = run_backtest(ticker.upper())

    # Create a trace for the ensemble strategy
    trace1 = go.Scatter(
        x=result.index,
        y=result['ensemble'],
        mode='lines',
        name='Ensemble Strategy'
    )

    # Create a trace for the Dow Jones Index
    trace2 = go.Scatter(
        x=result.index,
        y=result['dji'],
        mode='lines',
        name='Dow Jones Index'
    )

    # Create the layout
    layout = go.Layout(
        title=f"Backtesting Results: Ensemble Strategy vs Dow Jones Index for {ticker.upper()}",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        hovermode='closest'
    )

    # Create the figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()