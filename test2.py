
# %%
import os
# Import all required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# Import for splitting test and training data set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# %%
df = pd.read_csv('./covid-vaccination-vs-death_ratio.csv')

# %%
# define values
indonesia = ["Indonesia"]

# drop rows that contain any value in the list
df = df[df.country.isin(indonesia) == True]
df

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
minMAE

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

# holding 14 days that are looking back and 5 variables that we got
print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))  # 1 is the day after n_past

# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', input_shape=(
        trainX.shape[1], trainX.shape[2]), return_sequences=True),
    tf.keras.layers.LSTM(32, activation='relu', return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(trainY.shape[1])
])

optimizer = tf.keras.optimizers.SGD(lr=1.0000e-04, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

model.summary()

# %%
# Membuat fungsi callback


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('mae') < 0.1):
            print("\nmae kurang dari 10%")
            self.model.stop_training = True


callbacks = myCallback()

# %%
# Training Model
history = model.fit(trainX, trainY, batch_size=16, validation_split=0.2,
                    epochs=400, verbose=2, callbacks=[callbacks])

# %%
# Plot mae
plt.plot(history.history['mae'], label='Training mae')
plt.plot(history.history['val_mae'], label='Validation mae')
plt.legend()

# %%
# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()

# %%
# Start with the last day in training date and predict future
n_future = 248  # Redefining n_future to extend prediction dates beyond original n_future dates
forecast_period_dates = pd.date_range(
    list(sorted_data_graph['date'])[-30], periods=n_future, freq='1d').tolist()

print(forecast_period_dates)

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

print('5 Baris Teratas:\n', df_forecast.head())
print('\n5 Baris Terbawah:\n', df_forecast.tail())
print('\nShape Dataset:\n', df_forecast.shape)

# %%
original = sorted_data_graph[['date', 'ratio']]
original['date'] = pd.to_datetime(original['date'])
original = original.loc[original['date'] >= '2021-01-28']

print('5 Baris Teratas:\n', original.head())
print('\n5 Baris Terbawah:\n', original.tail())
print('\nShape Dataset:\n', original.shape)

# %%
fig, ax = plt.subplots(figsize=(15, 7))

sns.lineplot(ax=ax, x='date', y='ratio', data=original)
sns.lineplot(ax=ax, x='date', y='ratio', data=df_forecast)
plt.legend(labels=["Actual", "Prediction"])

# %%
model.save('ratio_vaccination.h5')