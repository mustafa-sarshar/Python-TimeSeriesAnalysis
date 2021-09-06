"""
Naive Forecast
"""
# In[] Libs
import pandas as pd
import matplotlib.pyplot as plt
from utils import metrics as met

# In[] Datasets
_dataset_dir = "GaitData"
_dataset_file = "01_LeftAnkle.csv"
df = pd.read_csv(f"..\\Datasets\\{_dataset_dir}\\{_dataset_file}", skiprows=12)
df.drop(columns=["PacketCounter", "StatusWord", "ClippingFlags", "RSSI"], inplace=True)

# In[] Plot the dataset
_cols_to_plot = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
plt.clf()
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# In[] Data Analysis
_col_to_workon = "Gyr_Z"
df[_col_to_workon+"_naive_forecast"] = df[_col_to_workon].shift(1)
y_true = df.iloc[1:][_col_to_workon]
y_pred = df.iloc[1:][_col_to_workon+"_naive_forecast"]

# In[] Plot the results
_cols_to_plot = [_col_to_workon, _col_to_workon+"_naive_forecast"]
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# In[] Calculate the metrics
print("SSE: {:0.2f}".format(met.metrics_sse(y_true, y_pred)))
print("MSE: {:0.2f}".format(met.metrics_mse(y_true, y_pred)))
print("RMSE: {:0.2f}".format(met.metrics_rmse(y_true, y_pred)))
print("MAE: {:0.2f}".format(met.metrics_mae(y_true, y_pred)))
print("MAPE: {:0.2f}".format(met.metrics_mape(y_true, y_pred)))
print("SMAPE: {:0.2f}".format(met.metrics_smape(y_true, y_pred)))
print("R2 Score: {:0.2f}".format(met.metrics_r2_score(y_true, y_pred)))

# In[] Finish