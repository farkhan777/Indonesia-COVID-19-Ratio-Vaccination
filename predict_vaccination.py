# %%
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import pandas as pd
import numpy as np


def predict():
    # %%# %%
    df = pd.read_csv('./covid-vaccination-vs-death_ratio.csv')

    # %%
    # define values
    indonesia = ["Indonesia"]

    # drop rows that contain any value in the list
    df = df[df.country.isin(indonesia) == True]

    # %%
    # Convert Data Types
    df['date'] = pd.to_datetime(df['date'])

    # %%
    # Print banyaknya vaksinasi di seluruh area Jawa Timur dari waktu ke waktu
    data_graph = df.groupby(['date', 'country'])['ratio'].max().reset_index()
    data_graph.sort_values(by="date")

    data_graph['date'] = pd.to_datetime(data_graph['date'])

    # %%
    sorted_data_graph = data_graph.sort_values(by='date', ascending=True)

    # %%
    minMAE = (sorted_data_graph['ratio'].max() -
              sorted_data_graph['ratio'].min()) * (10/100)

    # %%
    # Variables for training
    cols = list(sorted_data_graph)[2:7]

    df_cols = sorted_data_graph[cols].astype(float)

    # %%
    # LSTM used sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(df_cols)
    df_cols_scaled = scaler.transform(df_cols)

    # %%
    dates = sorted_data_graph['date'].values
    ratio = sorted_data_graph['ratio'].values

    # %%
    dates = np.asarray(dates).astype('float32')
    ratio = np.asarray(ratio).astype('float32')

    # %%
    # As required for LSTM networks, require to reshape an input data into n_samples x timesteps
    # In this example, the n_features is 2. I will make timesteps = 3
    # With this, the resultant n_samples is 5 (as the input data has 9 rows)
    trainX = []
    trainY = []

    n_future = 1  # Number of days I want to predict into the future
    n_past = 14  # Number of past days I want to use to predict the future

    for i in range(n_past, len(df_cols_scaled) - n_future+1):
        trainX.append(df_cols_scaled[i - n_past:i, 0:df_cols.shape[1]])
        trainY.append(df_cols_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)

    model = tf.keras.models.load_model('./ratio_vaccination.h5')

    # %%
    # Start with the last day in training date and predict future
    n_future = 248  # Redefining n_future to extend prediction dates beyond original n_future dates
    forecast_period_dates = pd.date_range(
        list(sorted_data_graph['date'])[-1], periods=n_future, freq='1d').tolist()

    # %%
    forecast = model.predict(trainX[-n_future:])

    # %%
    # Perform inverse transformation to rescale back to original range
    # Since we used 5 variables for transform, the inverse expects same dimensions
    # Therefore, just copy the values 5 times and discard them after inverse transform
    forecast_copies = np.repeat(forecast, df_cols.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]

    # %%
    # Convert timestamp to date
    forecast_dates = []
    for time_i in forecast_period_dates:
        forecast_dates.append(time_i.date())

    # %%
    df_forecast = pd.DataFrame(
        {'date': np.array(forecast_dates), 'ratio': y_pred_future})
    df_forecast['date'] = pd.to_datetime(df_forecast['date'])

    # %%
    original = sorted_data_graph[['date', 'ratio']]
    original['date'] = pd.to_datetime(original['date'])
    original = original.loc[original['date'] >= '2021-01-28']

    df_forecast['ratio'] = df_forecast['ratio'] + \
        original.loc[original.index[-1], "ratio"]

    return df_forecast, original
