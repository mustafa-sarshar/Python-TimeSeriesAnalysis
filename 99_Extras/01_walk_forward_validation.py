"""
Walk Forward Validation
"""
# In[] Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import itertools

# In[] Ignore all warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# In[] Define Methods
"""
This code is inspired by LazyProgrammer - Time Series Analysis, Forecasting, and Machine Learning (Udemy)
Link: https://www.udemy.com/share/104I9M3@mGTRIQjPGbtdhOVKI4L3Yel5StUCIzNJs5PET-BfPygzCNeUP9tI6PJjBoR-7g46/
"""
def walk_forward(
        dataframe,
        test_length,
        steps,
        trend_type,
        seasonal_type,
        damped_trend,
        init_method,
        use_boxcox,
        seasonal_periods,
        debug=False
):
    # Init params
    n_test = len(dataframe) - test_length - steps + 1
    
    # store errors
    errors = []
    seen_last = False
    steps_completed = 0
    
    for end_of_train_indx in range(n_test, len(dataframe) - test_length + 1):
        train = dataframe.iloc[:end_of_train_indx]
        test = dataframe.iloc[end_of_train_indx:end_of_train_indx + test_length]
    
        if test.index[-1] == dataframe.index[-1]: # To ensure that wir loop into the data entirely.
            seen_last = True
        
        steps_completed += 1
    
        hw = ExponentialSmoothing(
            train,
            initialization_method=init_method,
            trend=trend_type,
            damped_trend=damped_trend,
            seasonal=seasonal_type,
            seasonal_periods=seasonal_periods,
            use_boxcox=use_boxcox
        )
        res_hw = hw.fit()
    
        # compute error for the forecast horizon
        fcast = res_hw.forecast(test_length)
        error = mean_squared_error(test, fcast)
        errors.append(error)
    
    if debug:
        print("seen_last:", seen_last)
        print("steps completed:", steps_completed)
        print(f"Mean of Errors: {np.mean(errors):0.2f}")
    return np.mean(errors)    
    

# In[] Datasets
_dataset_dir = "GaitData"
_dataset_file = "02_LeftAnkle.csv"
df = pd.read_csv(f"..\\datasets\\{_dataset_dir}\\{_dataset_file}", skiprows=12)
df.drop(columns=["PacketCounter", "StatusWord", "ClippingFlags", "RSSI"], inplace=True)

# In[] Plot the dataset
_cols_to_plot = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
plt.clf()
plt.plot(df[_cols_to_plot])
plt.legend(_cols_to_plot)
plt.show()

# In[] Add new Params
df["FreeAcc_mag"] = (df["FreeAcc_E"]**2 + df["FreeAcc_N"]**2 + df["FreeAcc_U"]**2)**.5
df["Gyr_mag"] = (df["Gyr_X"]**2 + df["Gyr_Y"]**2 + df["Gyr_Z"]**2)**.5

# In[] Data Analysis - Smoothing and Fitting
_col_to_workon = "Gyr_Z"
dataframe = np.abs(df[_col_to_workon]) + 0.01
test_length = int(len(df) * 0.2)
steps = 10
trend_type = "add"
seasonal_type = "add"
damped_trend = False
init_method = "legacy-heuristic"
use_boxcox = 0

# In[] Test it for one manually given settings
walk_forward(dataframe, test_length, steps, trend_type, seasonal_type, damped_trend, init_method, use_boxcox, debug=True)

# In[] Test the model for different hyperparameters
trend_type_list = ["add", "mul"]
seasonal_type_list = ["add", "mul"]
damped_trend_list = [True, False]
init_method_list = ["estimated", "heuristic", "legacy-heuristic"]
use_boxcox_list = [True, False, 0]
seasonal_periods_list = [50, 100]

options_list = (
    trend_type_list,
    seasonal_type_list,
    damped_trend_list,
    init_method_list,
    use_boxcox_list,
    seasonal_periods_list,
)

options_total_list = []
for option in itertools.product(*options_list):
    print(option)
    options_total_list.append(option)
print(f"Total No. of options is: {len(options_total_list)}")
  
# In[]
cur_option = 1
best_score = float("inf")
best_options = None
print(f"Total No. of options is: {len(options_total_list)}")
for option in options_total_list:
    print(f"\nOption No. {cur_option}")
    try:
        score = walk_forward(dataframe, test_length, steps, *option, debug=True)
        
        if score < best_score:
            print(f"Best score so far: {score:0.2f}")
            best_score = score
            best_options = option
    except ValueError:
        print("Oops! This option may not work!!!")
    cur_option += 1
        
print(f"\nFinal best score: {best_score:0.2f}")
print("Final best options:")
print(f"\tStep: {best_options[0]}")
print(f"\tTrend Type: {best_options[1]}")
print(f"\tSeasonal Type: {best_options[2]}")
print(f"\tDamped Trend: {best_options[3]}")
print(f"\tInit Method: {best_options[4]}")
print(f"\tUse Boxcox: {best_options[5]}")
print(f"\tSeasonal Period: {best_options[6]}")

        