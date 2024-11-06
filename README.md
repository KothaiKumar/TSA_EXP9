# Developed By: Kothai K
# Reg No: 212222240051
# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from pmdarima import auto_arima

date_range = pd.date_range(start='2020-01-01', periods=1000, freq='D')
temperature_values = np.random.randint(15, 30, size=len(date_range))
weather_df = pd.DataFrame({'date': date_range, 'temperature': temperature_values})
weather_df.to_csv('weather_data.csv', index=False)

df = pd.read_csv('weather_data.csv', parse_dates=['date'], index_col='date')
print(df.head())
print(df.describe())

plt.figure(figsize=(12, 6))
plt.plot(df['temperature'])
plt.title('Time Series Plot of Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()

def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

adf_test(df['temperature'])

plt.figure(figsize=(12, 6))
plt.plot(df['temperature'])
plt.title('Time Series Plot of Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()
plot_acf(df['temperature'], lags=30)
plot_pacf(df['temperature'], lags=30)
plt.show()

df['temperature_diff'] = df['temperature'].diff().dropna()
plt.figure(figsize=(12, 6))
plt.plot(df['temperature_diff'])
plt.title('Differenced Time Series')
plt.xlabel('Date')
plt.ylabel('Differenced Temperature')
plt.show()

adf_test(df['temperature_diff'].dropna())

model = ARIMA(df['temperature'], order=(p, d, q))
model_fit = model.fit()

print(model_fit.summary())

forecast = model_fit.forecast(steps=10)
print(forecast)

plt.figure(figsize=(12, 6))
plt.plot(df['temperature'], label='Original Data')
plt.plot(forecast.index, forecast, color='red', label='Forecast', marker='o')
plt.title('ARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()
auto_model = auto_arima(df['temperature'], seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())


train_size = int(len(df) * 0.8)
train, test = df['temperature'][:train_size], df['temperature'][train_size:]

model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()
predictions = model_fit.forecast(steps=len(test))

# Calculate and print the Mean Squared Error
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual Data')
plt.plot(test.index, predictions, color='red', label='Predicted Data')
plt.title('Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/d480d719-59c0-4cbb-a1dd-e8cd4e3590c5)
![image](https://github.com/user-attachments/assets/0b519af1-d5d9-4aa8-b4bd-b729becc13f1)
![image](https://github.com/user-attachments/assets/269eadfa-cd24-4015-ae9e-dbc2ee8963fa)
![image](https://github.com/user-attachments/assets/afc54009-010e-4964-805e-30e2ca4a6da2)
![image](https://github.com/user-attachments/assets/d3c52559-8ef2-40ca-8f36-cc39ee7a0d79)
![image](https://github.com/user-attachments/assets/74033a94-bce6-4336-ac3b-26ff0f0d466f)


### RESULT:
Thus ,the program is run successfully based on the ARIMA model using python.
