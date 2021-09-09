"""
Probabilistic Model Selection
    Akaike Information Criterion (AIC). Derived from frequentist probability.
    Bayesian Information Criterion (BIC). Derived from Bayesian probability.
    Minimum Description Length (MDL). Derived from information theory.
    
Resource: https://machinelearningmastery.com/probabilistic-model-selection-measures/
"""

# In[] Libs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima as AARIMA
from utils.custom_functions import plot_the_predictions, plot_AutoARIMA

# In[] Datasets
_dataset_dir = "GaitData"
_dataset_file = "02_RightAnkle.csv"
df = pd.read_csv(f"..\\datasets\\{_dataset_dir}\\{_dataset_file}", skiprows=12)
df.drop(columns=["PacketCounter", "StatusWord", "ClippingFlags", "RSSI"], inplace=True)
df = df[1300:2150] # shrink the size of the dataframe

# In[] Plot the dataset
_cols_to_plot = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
plt.clf()
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# In[] Train and Test the model by splitting the data set into Train and Test sets
_col_to_workon = "Gyr_Z"
_train_test_ratio = 0.8 # equals to 80%
_n_cutoff = int(len(df) * _train_test_ratio)
_n_test = len(df) - _n_cutoff
df_train = df[_col_to_workon].iloc[:_n_cutoff]
df_test = df[_col_to_workon].iloc[_n_cutoff:]

arima_model = AARIMA(
    y=df_train,
    trace=True,
    suppress_warnings=True,
)

print(arima_model.summary())

# In[]
test_pred, confint = arima_model.predict(n_periods=_n_test, return_conf_int=True)
df[_col_to_workon+"_AARIMA"] = np.concatenate((df_train.values, test_pred))

# In[] Plot the predictions
_cols_to_plot = [_col_to_workon, _col_to_workon+"_AARIMA"]
plt.clf()
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# In[] Plot the predictions
plot_the_predictions(df_main=df_test.values, df_pred=test_pred, conf_int=confint)

# In[] Plot the whole data + the predictions
plot_AutoARIMA(model=arima_model, df_main=df[_col_to_workon], df_test=df_test)

# In[] Finish
