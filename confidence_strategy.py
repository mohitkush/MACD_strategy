
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import talib
from pathlib import Path
import datetime
from time import time
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import talib as ta
from finta import TA
import math
import sys
from scipy.signal import argrelextrema
from config import config
from sklearn.preprocessing import MinMaxScaler


# grouping the data as required
def create_df(df, freq):
    result = df.groupby([pd.Grouper(key='date_time', freq=freq), pd.Grouper(key='ticker')]) \
        .agg(Open=pd.NamedAgg(column='price', aggfunc='first'),
             Close=pd.NamedAgg(column='price', aggfunc='last'),
             High=pd.NamedAgg(column='price', aggfunc='max'),
             Low=pd.NamedAgg(column='price', aggfunc='min'),
             Volume=pd.NamedAgg(column='volume', aggfunc='sum')) \
        .reset_index()
    return result


def binary(x):
    if x:
        return 1
    else:
        return 0


def calculate_macd_adj_and_consolidation(df):
    DUMMY_VAL = 50
    dummy_series = pd.Series(np.full(DUMMY_VAL, df.Close.iloc[0]))
    macd, macdsignal, macdhist = ta.MACD(pd.concat([dummy_series, df.Close]), fastperiod=12, slowperiod=26,
                                         signalperiod=9)

    # macd, macdsignal, macdhist = ta.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)

    macd = macd[DUMMY_VAL:]
    macdsignal = macdsignal[DUMMY_VAL:]
    macdhist = macdhist[DUMMY_VAL:]
    df["MACD"] = macd
    df["MACDSIGNAL"] = macdsignal
    df["MACDHIST"] = macdhist
    df["MACDDIFF"] = df["MACD"] - df["MACDSIGNAL"]
    df["MACDHISTTHRES"] = 0
    df.loc[(df["MACDHIST"] > df["MACDHIST"].quantile(q=0.4)) & (
                df["MACDHIST"] < df["MACDHIST"].quantile(q=0.6)), 'MACDHISTTHRES'] = 1
    df['macdsignalisgtmacd'] = df['MACDSIGNAL'] > df['MACD']
    df['macdcross'] = df['macdsignalisgtmacd'].rolling(window=2).apply(
        lambda x: 0 if x[0] == x[1] else (1 if x[0] else -1))
    df['count_index'] = range(0, len(df))

    df.reset_index(level=1, inplace=True)
    macd_cross_indexes = df.index
    DUMMY_VAL = 50

    df["MACD_ADJ"] = np.nan
    df["MACDSIGNAL_ADJ"] = np.nan
    df["MACDHIST_ADJ"] = np.nan

    macd_cross_indexes = df[df["macdcross"] != 0].count_index.array
    # df.reset_index(level=1, inplace=True)
    refrence_time = df.index.min()

    earlier_idx = 0
    for index, macd_cross_index in enumerate(macd_cross_indexes):

        if macd_cross_index == 0:
            continue
        dummy_series = pd.Series(np.full(DUMMY_VAL, df.Close.iloc[macd_cross_index]))
        last_val = index + 1
        if macd_cross_index > len(macd_cross_indexes):
            last_val = len(macd_cross_indexes) - 1
        macd, macdsignal, macdhist = ta.MACD(pd.concat([dummy_series,
                    df.Close.iloc[macd_cross_index:macd_cross_indexes[last_val]]]), fastperiod=12, slowperiod=26,
                                             signalperiod=9)
        macd = macd[DUMMY_VAL:]
        macdsignal = macdsignal[DUMMY_VAL:]
        macdhist = macdhist[DUMMY_VAL:]

        df.loc[refrence_time + pd.to_timedelta("%d minutes" % macd_cross_index): refrence_time + pd.to_timedelta(
            "%d minutes" % macd_cross_indexes[last_val]), 'MACD_ADJ'] = macd
        df.loc[refrence_time + pd.to_timedelta("%d minutes" % macd_cross_index): refrence_time + pd.to_timedelta(
            "%d minutes" % macd_cross_indexes[last_val]), 'MACDSIGNAL_ADJ'] = macdsignal
        df.loc[refrence_time + pd.to_timedelta("%d minutes" % macd_cross_index): refrence_time + pd.to_timedelta(
            "%d minutes" % macd_cross_indexes[last_val]), 'MACDHIST_ADJ'] = macdhist

        earlier_idx = macd_cross_index

    # RSI
    # Create a variable n with a value of 10
    n = 14
    rsi_smooth_period = 7
    # Create a column by name, RSI and assign the calculation of RSI to it
    df['RSI'] = ta.RSI(df['Close'], timeperiod=n)

    # HMA smoothening over RSI
    df_rsi = pd.DataFrame({"Close": df['RSI'], "Open": df['RSI'], "High": df['RSI'], "Low": df['RSI']})
    # RSI smooth is smoothened RSI over 5 period
    df["RSI_HMA"] = TA.HMA(df_rsi, rsi_smooth_period)

    # df["RSI_HMA"] = TA.HMA(df_rsi, rsi_smooth_period)

    # RSI smooth is smoothened RSI over 5 period
    df["HMA"] = TA.HMA(df, 9)
    slopes = ta.LINEARREG_SLOPE(df['Close'], timeperiod=9)
    # normal argrelextrema function
    df['HMA_MIN'] = df.iloc[argrelextrema(df['HMA'].values, np.less, order=2)[0]]['HMA']
    df['HMA_MAX'] = df.iloc[argrelextrema(df['HMA'].values, np.greater, order=2)[0]]['HMA']
    df['HMA_MAX_F'] = df['HMA_MAX'].ffill(axis=0)
    df['HMA_MIN_F'] = df['HMA_MIN'].ffill(axis=0)
    con1 = (df['HMA'] - df['HMA_MIN_F'] > -1 * df['HMA'] * 0.001) & (df['HMA'] - df['HMA_MAX_F'] < df['HMA'] * 0.001)
    con2 = (df['HMA_MAX_F'] - df['HMA_MIN_F']) < df['HMA'] * 0.004
    con3 = con1 & con2
    df['is_con'] = con3
    df['is_con'] = df['is_con'].apply(binary)
    df['MAX'] = df['HMA'] * df['is_con']
    df['MAX'].replace(0, np.nan, inplace=True)
    df.drop(['MACDHIST', 'MACDDIFF', 'MACDHISTTHRES',
             'macdsignalisgtmacd', 'macdcross', 'count_index', 'RSI', 'RSI_HMA', 'HMA', 'HMA_MIN',
             'HMA_MAX', 'HMA_MAX_F', 'HMA_MIN_F', 'MAX'], axis=1, inplace=True)

    return df


# method to compute trades/day and returns/day for a single stock
def calculate_trades(fdf):

    trades = []
    index = []
    # fdf.reset_index(level=0, inplace=True)
    bt = Backtest(fdf, MACDCrossover, cash=100000, commission=0,
                  exclusive_orders=False, trade_on_close=False)
    stats = bt.run()
    stats = pd.Series(bt.run())
    stats['ticker'] = fdf['ticker'].unique()[0]
    index.append([fdf['ticker'].unique(), fdf['date'].dt.date.unique()])
    return stats['_trades'], stats, index


def is_extrema(df):
    df['is_max'] = df.rolling(15, min_periods=1)['MACD_ADJ'].apply(lambda x: 1 if x[-1] >= x.nlargest(4).min() else 0)
    df['is_min'] = df.rolling(15, min_periods=1)['MACD_ADJ'].apply(lambda x: 1 if x[-1] <= x.nsmallest(4).max() else 0)
    df['is_max'] = df.groupby(df.is_max.diff().ne(0).cumsum()).is_max.cumsum()
    df['is_min'] = df.groupby(df.is_min.diff().ne(0).cumsum()).is_min.cumsum()
    return df


def last_crossover_macd(df):
    df['macdsignalisgtmacd'] = df['MACDSIGNAL_ADJ'] > df['MACD_ADJ']
    df['macdcross'] = df['macdsignalisgtmacd'].rolling(window=2).apply(
        lambda x: 0 if x[0] == x[1] else (1 if x[0] else -1))

    df['count_index'] = range(0, len(df))

    df['Next_Timestamp'] = df[df.macdcross != 0].count_index
    df['Next_Timestamp'].fillna(method='ffill', inplace=True)

    df['macdcrosssince'] = df['count_index'] - df['Next_Timestamp']
    df['macdcrosssince'] = np.where(df['MACD_ADJ'].isnull(), np.nan, df['macdcrosssince'])

    df.drop(['macdsignalisgtmacd', 'macdcross', 'Next_Timestamp'], axis=1, inplace=True)

    return df


def confidence_score(df):
    sum = 0
    # hma_smooth_period = 8
    # df_hma = pd.DataFrame({"Close": df['Close'], "Open": df['Close'], "High": df['Close'], "Low": df['Close']})
    # df['HMA'] = TA.HMA(df_hma, hma_smooth_period)
    df['macd_1'] = df['MACD'] - df['MACD'].shift(1)

    for i in range(2, 9):
        df[f'macd_{i}'] = df['macd_1'].shift(i - 1)

    df['confidence'] = 0
    df['confidence'] = np.where(df.macdcrosssince == 1, df['macd_1'] * config.weights_config['1']['1'],
                                df['confidence'])
    df['confidence'] = np.where(df.macdcrosssince == 2, df['macd_1'] * config.weights_config['2']['1'] + df['macd_2'] *
                                config.weights_config['2']['2'], df['confidence'])
    df['confidence'] = np.where(df.macdcrosssince == 3, df['macd_1'] * config.weights_config['3']['1'] + df['macd_2'] *
                                config.weights_config['3']['2'] + df['macd_3'] * config.weights_config['3']['3'],
                                df['confidence'])
    df['confidence'] = np.where(df.macdcrosssince == 4, df['macd_1'] * config.weights_config['4']['1'] + df['macd_2'] *
                                config.weights_config['4']['2'] + df['macd_3'] * config.weights_config['4']['3'] + df[
                                    'macd_4'] * config.weights_config['4']['4'], df['confidence'])
    df['confidence'] = np.where(df.macdcrosssince == 5, df['macd_1'] * config.weights_config['5']['1'] + df['macd_2'] *
                                config.weights_config['5']['2'] + df['macd_3'] * config.weights_config['5']['3'] + df[
                                    'macd_4'] * config.weights_config['5']['4'] + df['macd_5'] *
                                config.weights_config['5']['5'], df['confidence'])
    df['confidence'] = np.where(df.macdcrosssince == 6, df['macd_1'] * config.weights_config['6']['1'] + df['macd_2'] *
                                config.weights_config['6']['2'] + df['macd_3'] * config.weights_config['6']['3'] + df[
                                    'macd_4'] * config.weights_config['6']['4'] + df['macd_5'] *
                                config.weights_config['6']['5'] + df['macd_6'] * config.weights_config['6']['6'],
                                df['confidence'])
    df['confidence'] = np.where(df.macdcrosssince == 7, df['macd_1'] * config.weights_config['7']['1'] + df['macd_2'] *
                                config.weights_config['7']['2'] + df['macd_3'] * config.weights_config['7']['3'] + df[
                                    'macd_4'] * config.weights_config['7']['4'] + df['macd_5'] *
                                config.weights_config['7']['5'] + df['macd_6'] * config.weights_config['7']['6'] + df[
                                    'macd_7'] * config.weights_config['7']['7'], df['confidence'])
    df['confidence'] = np.where(df.macdcrosssince >= 8, df['macd_1'] * config.weights_config['8']['1'] + df['macd_2'] *
                                config.weights_config['8']['2'] + df['macd_3'] * config.weights_config['8']['3'] + df[
                                    'macd_4'] * config.weights_config['8']['4'] + df['macd_5'] *
                                config.weights_config['8']['5'] + df['macd_6'] * config.weights_config['8']['6'] + df[
                                    'macd_7'] * config.weights_config['8']['7'] + df['macd_8'] *
                                config.weights_config['8']['8'], df['confidence'])

    for i in range(1, 9):
        df.drop([f'macd_{i}'], inplace=True, axis=1)

    df['confidence'] = df['confidence'].fillna(0)

    df['confidence'] = np.where(df['is_con'], 0, df['confidence'])

    scaler_p = MinMaxScaler()
    df_p = df[df['confidence'] >= 0]
    df_p['confidence_scaled'] = scaler_p.fit_transform(df_p['confidence'].to_numpy().reshape(-1, 1))

    scaler_n = MinMaxScaler(feature_range=(-1, 0))
    df_n = df[df['confidence'] < 0]
    df_n['confidence_scaled'] = scaler_n.fit_transform(df_n['confidence'].to_numpy().reshape(-1, 1))

    df = df_n.append(df_p)
    df.sort_index(ascending=True, level=1, axis=0, inplace=True)
    df.drop(['confidence', 'macdcrosssince'], inplace=True, axis=1)
    df.rename(columns={'confidence_scaled': 'confidence'}, inplace=True)

    return df


# strategy class
# again grouping the data with new columns of macd and signal
class MACDCrossover(Strategy):

    def init(self):
        price = self.data.Close
        # self.ma1 = self.I(SMA, price, 10)
        # self.ma2 = self.I(SMA, price, 20)
        # self.macd, self.signal, self.hist = self.I(ta.MACD, price, 12, 26, 9)

    def next(self):

        config = {
            'max_trade_time': pd.Timedelta('29 minutes'),
            'initial_size': 0.1,
            'increased_size': 0.2,
            'stop_loss': None,
            'target': None,
            'macd_threshold': 0.1,
        }
        stop_loss_congif = {
            '2min': 0.2,
            '5min': 0.3,
            '15min': 0.35,
            '30min': 0.4
        }
        confidence_config = {
            '0min': 0.4,
            '4min': 0.6
        }

        current_time = self.data.index[-1]
        current_price = self.data.Close[-1]

        if current_time.time() > pd.Timestamp(2017, 1, 1, 9, 30).time():

            # closing trade if active duration gets above 30 minutes
            for trade in self.trades:
                if current_time - trade.entry_time >= config['max_trade_time']:
                    trade.close()

                if (current_time - trade.entry_time).total_seconds()/60 == 4 and self.data.confidence[-1] > \
                        confidence_config['4min']:
                    trade.buy(size=config['increased_size'])

                if (current_time - trade.entry_time).total_seconds()/60 == 4 and self.data.confidence[-1] < \
                        confidence_config['4min']*-1:
                    trade.sell(size=config['increased_size'])

                if trade.is_long:
                    if pd.Timedelta('3 minutes') > current_time - trade.entry_time and \
                            current_price <= (trade.max-trade.entry_price)*stop_loss_congif['2min']+trade.entry_price*0.99:
                        trade.close()
                    if pd.Timedelta('3 minutes') <= current_time - trade.entry_time <= pd.Timedelta('5 minutes') and \
                            current_price < (trade.max-trade.entry_price)*stop_loss_congif['5min']+trade.entry_price:
                        trade.close()
                    elif pd.Timedelta('5 minutes') < current_time - trade.entry_time <= pd.Timedelta('15 minutes') and \
                            current_price < (trade.max-trade.entry_price)*stop_loss_congif['15min']+trade.entry_price:
                        trade.close()
                    elif pd.Timedelta('15 minutes') < current_time - trade.entry_time < pd.Timedelta('30 minutes') and \
                            current_price < (trade.max-trade.entry_price)*stop_loss_congif['30min']+trade.entry_price:
                        trade.close()

                    if self.data.confidence[-5] > self.data.confidence[-4] > self.data.confidence[-3] > self.data.confidence[-2] > self.data.confidence[-1]:
                        trade.close()

                if trade.is_short:
                    if pd.Timedelta('3 minutes') > current_time - trade.entry_time and \
                            current_price >= trade.entry_price*1.01 - (trade.entry_price - trade.min)*stop_loss_congif['2min']:
                        trade.close()
                    if pd.Timedelta('3 minutes') <= current_time - trade.entry_time <= pd.Timedelta('5 minutes') and \
                            current_price > trade.entry_price - (trade.entry_price - trade.min)*stop_loss_congif['5min']:
                        trade.close()
                    elif pd.Timedelta('5 minutes') < current_time - trade.entry_time <= pd.Timedelta('15 minutes') and \
                            current_price > trade.entry_price - (trade.entry_price - trade.min)*stop_loss_congif['15min']:
                        trade.close()
                    elif pd.Timedelta('15 minutes') < current_time - trade.entry_time <= pd.Timedelta('30 minutes') and \
                            current_price > trade.entry_price - (trade.entry_price - trade.min)*stop_loss_congif['15min']:
                        trade.close()

                    if self.data.confidence[-5] < self.data.confidence[-4] < self.data.confidence[-3] < self.data.confidence[-2] < self.data.confidence[-1]:
                        trade.close()

            if self.data.confidence[-2] >= confidence_config['0min']:
                if not self.position.is_long:
                    self.buy(size=config['initial_size'])

            if self.data.confidence[-1] <= confidence_config['0min'] * -1:
                if not self.position.is_short:
                    self.sell(size=config['initial_size'])

            # try:
            #     if self.data.confidence[-3] < self.data.confidence[-2] < self.data.confidence[-1]:
            #         if not self.position.is_long:
            #             self.buy(size=config['initial_size'])
            #
            #     if self.data.confidence[-3] > self.data.confidence[-2] > self.data.confidence[-1]:
            #         if not self.position.is_short:
            #             self.sell(size=config['initial_size'])
            #
            # except IndexError:
            #     pass


df_ = pd.read_csv("../data/market_22_Jun_csv.csv", parse_dates=True, dtype={1: str, 2: float, 3: float}, sep=",",
                 header=None)
df_23 = pd.read_csv("../data/market_23_Jun_csv.csv", parse_dates=True, dtype={1: str, 2: float, 3: float}, sep=",",
               header=None)
df_24 = pd.read_csv("../data/market_24_Jun_csv.csv", parse_dates=True, dtype={1: str, 2: float, 3: float}, sep=",",
               header=None)
df_25 = pd.read_csv("../data/market_25_Jun_csv.csv", parse_dates=True, dtype={1: str, 2: float, 3: float}, sep=",",
               header=None)
df_ = df_.append([df_23, df_24, df_25], ignore_index=True)
df_.columns = ['date_time', 'ticker', 'price', 'volume']
df_.dropna(inplace=True)

df_['date_time'] = pd.to_datetime(df_['date_time'], format='%b %d %H:%M:%S %Y')
df_['date'] = pd.to_datetime(df_['date_time'].dt.date)
df_['price'] = df_['price']/100
df_ = df_[df_['date_time'].dt.time.between(datetime.time(9, 15, 0), datetime.time(15, 30, 0))]


# stocks from command line argument

# for i in range(1, len(sys.argv)):
#     stocks.append(sys.argv[i])

df = pd.DataFrame()
for ticker, date in config.stock_day_pair:
    df = df.append(df_[(df_['ticker'] == ticker) & (df_['date'] == date)])

df = create_df(df, '1T')
# df.set_index(['date_time'], inplace=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df['date'] = pd.to_datetime(df['date_time'].dt.date)


df.sort_values(['date_time'], inplace=True)
df.set_index(['date_time', 'ticker'], inplace=True)
# grouping data using ticker and date

if len(df['date'].unique()) == 1 and len(df.index.get_level_values('ticker').unique()) == 1:
    df = calculate_macd_adj_and_consolidation(df)
    df = is_extrema(df)
    df = last_crossover_macd(df)
    df = confidence_score(df)
    df.sort_values(by=['ticker', 'date_time'], inplace=True)


else:
    # by_ticker = df.groupby(['ticker'], group_keys=False)
    by_ticker_date = df.sort_index().groupby(['ticker', 'date'], group_keys=False)
    df = by_ticker_date.apply(calculate_macd_adj_and_consolidation)
    by_ticker_date = df.sort_index().groupby(['ticker', 'date'], group_keys=False)
    df = by_ticker_date.apply(is_extrema)
    by_ticker_date = df.sort_index().groupby(['ticker', 'date'], group_keys=False)
    df = by_ticker_date.apply(last_crossover_macd)
    by_ticker_date = df.sort_index().groupby(['ticker', 'date'], group_keys=False)
    df = by_ticker_date.apply(confidence_score)
    df.sort_values(by=['ticker', 'date_time'], inplace=True)


# df = confidence_score(df)
df.to_csv('../results/sample.csv')
by_ticker_date = df.sort_index().groupby(['ticker', 'date'], group_keys=False)
t = by_ticker_date.apply(calculate_trades)

index_ = []
for i in t:
    for index in i[2]:
        index_.append(index)

returns_ = pd.DataFrame(index_, columns=['ticker', 'date'])

trades = pd.DataFrame()
returns = pd.DataFrame()
for trade, stats, index in t:
    returns = returns.append(stats, ignore_index=True)
    trade['ticker'] = stats['ticker']
    trades = trades.append(trade, ignore_index=True)

returns_ = pd.concat([returns_, returns], axis=1)
returns_.drop(['_equity_curve', '_trades'], axis=1, inplace=True)
avg_ret = (returns['Return [%]']).sum()/returns_.shape[0]
print(avg_ret)
trades['date'] = trades.EntryTime.dt.date
trades.to_csv('../results/trades.csv', index=False)
returns_.to_csv('../results/returns.csv', index=False)
