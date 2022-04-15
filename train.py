import os
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

### Create the Stacked LSTM algorithm
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, ConvLSTM2D, Input
from tensorflow.keras import backend as K
K.clear_session()

csv = r'Dataset/USDJPY_mt5_ticks.csv'
curr1 = csv.split('_')[0].split('.')[0]
currency = curr1.split('/')[1]

interval = '240Min'
epoch = 4

if not os.path.exists('trained_models'):
    os.mkdir("trained_models")

if not os.path.exists('trained_models' + os.sep + currency):
    os.mkdir('trained_models' + os.sep + currency)

if not os.path.exists('trained_models' + os.sep + currency + os.sep + interval):
    os.mkdir('trained_models' + os.sep + currency + os.sep + interval)

high_path = 'trained_models' + os.sep + currency + os.sep + interval + os.sep + 'high_model'
if not os.path.exists(high_path):
    os.mkdir(high_path)


low_path = 'trained_models' + os.sep + currency + os.sep + interval + os.sep + 'low_model'
if not os.path.exists(low_path):
    os.mkdir(low_path)


file_name = csv.split('/')[1].split('.')[0]
print("CURRENCY: ", file_name)


def model_train(x_train, y_train, save_name):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    # model.add(Dropout(0.1))
    model.add(Dropout(0.5))  # newly added by zhr
    model.add(LSTM(50, return_sequences=True))  # newly added by zhr
    # model.add(Dropout(0.5))
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(0.1))
    # model.add(LSTM(64, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(50))
    # model.add(Dropout(0.1))
    model.add(Dense(units=y_train.shape[1]))
    # model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=64, verbose=1, validation_split=0.2)
    model.summary()
    model.save(save_name)

    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()
    # plt.show()
    return model


def get_forecast_df(model, x_train, col_ind, col_name):
    forecast = model.predict(x_train[-n_future:])  # forecast
    # Perform inverse transformation to rescale back to original range
    # Since we used 5 variables for transform, the inverse expects same dimensions
    # Therefore, let us copy our values 5 times and discard them after inverse transform
    forecast_copies = np.repeat(forecast, train_set.shape[1], axis=-1)
    y_pred_future = sc.inverse_transform(forecast_copies)[:, col_ind]

    forecast_dates = []
    for time_i in forecast_period_dates:
        forecast_dates.append(time_i)

    df_forecast = pd.DataFrame({'DateTime': np.array(forecast_dates), col_name: y_pred_future})
    df_forecast['DateTime'] = pd.to_datetime(df_forecast['DateTime'])
    # df_forecast.to_csv("df_forecast.csv")
    # print(df_forecast.shape)
    return df_forecast


def get_rsi(file, value, n):
    """
    calculates -> RSI value
    takes argument -> dataframe, column name, period value
    returns dataframe by adding column : 'RSI_' + column name
    """
    delta = file[value].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(span=n, adjust=False).mean()
    ema_down = down.ewm(span=n, adjust=False).mean()
    rs = ema_up / ema_down
    file['RSI_' + value] = 100 - (100 / (1 + rs))

    return file


def moving_avg(ultratech_df, value, fast_p, slow_p):
    """
    calculates -> slow moving average, fast moving average
    takes argument -> dataframe, column name, slow period, fast period
    returns dataframe by adding columns -> 'MA_Slow_HLCC/4',  'SMA_period', MA_Fast_HLCC/4', 'FMA_period'
    """

    ultratech_df['MA_Slow_HLCC/4'] = ultratech_df[value].rolling(window=17, min_periods=1).mean()
    ultratech_df['SMA_period'] = slow_p
    ultratech_df['MA_Fast_HLCC/4'] = ultratech_df[value].rolling(window=7, min_periods=1).mean()
    ultratech_df['FMA_period'] = fast_p

    return ultratech_df


file_path = r'dataset_currency_data_ohlc_rsi'
df = pd.read_csv(csv)
print(len(df))

df['DateTime'] = pd.to_datetime(df['DateTime'])

df = df.set_index(df['DateTime'])

data_bid = df['Bid'].resample(interval).ohlc()
data_ask = df['Ask'].resample(interval).ohlc()

data = pd.DataFrame()
#
data['open'] = (data_ask['open'] + data_bid['open']) / 2
data['high'] = (data_ask['high'] + data_bid['high']) / 2
data['low'] = (data_ask['low'] + data_bid['low']) / 2
data['close'] = (data_ask['close'] + data_bid['close']) / 2
# data = data.reset_index()

# check_for_nan = data['open'].isnull().values.any()
# print(check_for_nan)

updated_df = data
updated_df['open'] = updated_df['open'].fillna(updated_df['open'].mean())
updated_df['high'] = updated_df['high'].fillna(updated_df['high'].mean())
updated_df['low'] = updated_df['low'].fillna(updated_df['low'].mean())
updated_df['close'] = updated_df['close'].fillna(updated_df['close'].mean())

# updated_df = updated_df.reset_index()
updated_df['HLCC/4'] = (updated_df['high'] + updated_df['low'] + updated_df['close'] + updated_df['close']) / 4
updated_df = get_rsi(updated_df, 'HLCC/4', 14)
updated_df = moving_avg(updated_df, 'HLCC/4', 17, 7)
updated_csv_path = file_path + os.sep + f"{file_name}_ohlc" + '.csv'
updated_df.to_csv(updated_csv_path)

high_model_save_path =  high_path + os.sep + 'high.h5'
low_model_save_path =  low_path + os.sep + 'low.h5'

df = pd.read_csv(updated_csv_path)
df = df.drop(range(0, 200))

# print(df.head())

cols = ['high', 'low', 'RSI_HLCC/4', 'MA_Slow_HLCC/4', 'MA_Fast_HLCC/4']

train_set = df[cols].astype(float)
print("train_set", train_set)
train_dates = pd.to_datetime(df['DateTime'])

sc = StandardScaler()
scaled_data = sc.fit_transform(train_set)

n_past = 48
n_future = 1
high_x_train = []
high_y_train = []

for x in range(n_past, len(scaled_data) - n_future + 1):
    high_x_train.append(scaled_data[x - n_past:x, 0:scaled_data.shape[1]])
    high_y_train.append(scaled_data[x + n_future - 1:x + n_future, 0])

low_x_train = []
low_y_train = []

for x in range(n_past, len(scaled_data) - n_future + 1):
    low_x_train.append(scaled_data[x - n_past:x, 0:scaled_data.shape[1]])
    low_y_train.append(scaled_data[x + n_future - 1:x + n_future, 1])

high_x_train, high_y_train = np.array(high_x_train), np.array(high_y_train)
print(high_x_train.shape)
print(high_y_train.shape)

low_x_train, low_y_train = np.array(low_x_train), np.array(low_y_train)
print(low_x_train.shape)
print(low_y_train.shape)

"""
training of models
"""
high_model = model_train(
    high_x_train,
    high_y_train,
    high_model_save_path)

low_model = model_train(
    low_x_train,
    low_y_train,
    low_model_save_path)

"""
loading saved model
"""
high_new_model_path = high_model_save_path
high_new_model = load_model(high_new_model_path)

low_new_model_path = low_model_save_path
low_new_model = load_model(low_new_model_path)

# Forecasting...
# Start with the last day in training date and predict future...
n_future = 10  # Redefining n_future to extend prediction dates beyond original n_future dates...
print(list(train_dates)[-1])
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future,
                                      freq=interval.lower()).tolist()  # H - Hour, m - min
print(forecast_period_dates)
high_df_forecast = get_forecast_df(high_new_model, high_x_train, 0, 'high')
print("high")
print(high_df_forecast)
# high_df_forecast.to_csv("high_df_pred.csv.csv")
low_df_forecast = get_forecast_df(low_new_model, low_x_train, 1, 'low')
print("low")
print(low_df_forecast)
