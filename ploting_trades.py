import plotly.graph_objects as go
import talib as ta
import pandas as pd
import datetime
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from config import config


trades = pd.read_csv('../results/trades.csv', parse_dates=['date'])

df = pd.read_csv('../data/sample.csv')
df['date_time'] = pd.to_datetime(df['date_time'])
df['date'] = pd.to_datetime(df['date_time'].dt.date)
df.set_index(['ticker', 'date_time'], inplace=True)


def chart_strategy(df, trades):

    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 close=df['Close'],
                                 low=df['Low'],
                                 high=df['High']))

    for i in trades.index:
        if trades['EntryPrice'][i] - trades['ExitPrice'][i] > 0:
            fig.add_trace(
                go.Scatter(
                    x=[trades['ExitTime'][i]],
                    y=[trades['ExitPrice'][i] + 30],
                    mode="markers+text",
                    marker=dict(symbol='triangle-up-open', size=12),
                    # text = 'important',
                    # textposition = 'middle right'
                )
            )
        elif trades['EntryPrice'][i] - trades['ExitPrice'][i] < 0:
            fig.add_trace(
                go.Scatter(
                    x=[trades['ExitTime'][i]],
                    y=[trades['ExitPrice'][i] + 30],
                    mode="markers+text",
                    marker=dict(symbol='triangle-down-open', size=12),
                    # text = 'important',
                    # textposition = 'middle right'
                )
            )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACDSIGNAL_ADJ'], yaxis='y2', name='signal', line=dict(color='green', width=.8)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_ADJ'], yaxis='y2', name='macd', line=dict(color='royalblue', width=.8)))

    fig.add_trace(go.Scatter(x=df.index, y=df['confidence'], yaxis='y2', name='confidence_score', line=dict(color='purple', width=.8)))

    ticker = df['ticker'].unique()[0]
    date = np.datetime_as_string(trades['date'].unique()[0], unit='D')
    titles = (f'Stock id : {ticker}',
              f'Date : {date}')
    # Add titles
    fig.update_layout(
        title=str(titles),
        yaxis_title='Stock Price (INR per Shares)')

    # fig.layout.yaxis2.showgrid = False

    mins_macd = df['MACD_ADJ'].min()
    maxs_macd = df['MACD_ADJ'].max()
    # min_macd = df['RSI_HMA'].min()
    # max_macd = df['RSI_HMA'].max()

    macd_val = np.ceil(np.max(np.absolute(np.array([mins_macd, maxs_macd]))))

    fig.update_yaxes(title_text="<b>secondary</b> MACD", range=[-1 * macd_val, 1 * macd_val], secondary_y=True,
                     anchor="free", position=1)

    fig.update_xaxes(rangeslider_thickness=0.05)
    # fig.update_layout(width=900, height=900)

    return fig


for ticker, date in config.stock_day_pair:
    df_ = df[(df.index.get_level_values(0) == int(ticker)) & (df.date == date)]
    trades_ = trades[(trades['ticker'] == int(ticker)) & (trades['date'] == date)]
    if len(trades_) == 0:
        continue
    df_.reset_index(level=0, inplace=True)
    # df.sort_index(inplace=True)
    chart_strategy(df_, trades_).show()

# chart_strategy(df, trades).show()
