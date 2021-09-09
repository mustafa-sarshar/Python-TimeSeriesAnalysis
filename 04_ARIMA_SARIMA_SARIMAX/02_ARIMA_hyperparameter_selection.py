"""
Partial Autocorrelation Function (PACF) - to select p for AR(p)
    Set p = maximum non-zero (outside confidence interval) lag 
Autocorrelation Function (ACF) - to find q for MA(q)
    Set q = maximum non-zero (outside confidence interval) lag
"""
# In[] Libs
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# In[] Datasets
_dataset_dir = "GaitData"
_dataset_file = "02_RightAnkle.csv"
df = pd.read_csv(f"..\\datasets\\{_dataset_dir}\\{_dataset_file}", skiprows=12)
df.drop(columns=["PacketCounter", "StatusWord", "ClippingFlags", "RSSI"], inplace=True)

# In[] Plot the dataset
_cols_to_plot = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
plt.clf()
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# In[] Plot the PACF and ACF functions to determine p and q hyperparameters
_col_to_workon = "Gyr_Z"
fig, axes = plt.subplots(2, 1)
plot_pacf(x=df[_col_to_workon], ax=axes[0])
plot_acf(x=df[_col_to_workon], ax=axes[1])
plt.show()

# In[] Finish