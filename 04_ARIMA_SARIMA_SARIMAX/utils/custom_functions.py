def plot_fit_and_forecast(
        df_main,
        df_train,
        df_test,
        forecast_model
):
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.clf()
    plt.plot(df_main, label="org. data")
    
    # plot the curve fitted on train set
    train_pred = forecast_model.fittedvalues
    plt.plot(train_pred[df_train.index], label="fitted values")
    
    # forecast the test set
    prediction_result = forecast_model.get_forecast(len(df_test))
    conf_int = prediction_result.conf_int()
    lower, upper = conf_int.iloc[:, 0], conf_int.iloc[:, 1]
    forecast = prediction_result.predicted_mean
    plt.plot(df_test.index, forecast, label="Forecast")
    plt.fill_between(df_test.index, lower, upper, color="red", alpha=0.3)
    plt.title(f"Avg. lower: {np.mean(lower):0.2f} ---> Avg. upper: {np.mean(upper):0.2f}")
    plt.legend()
    plt.show()
    

def plot_the_predictions(
        df_main,
        df_pred,
        conf_int
):
    import matplotlib.pyplot as plt
    
    plt.clf()
    plt.plot(df_main, label="org. data")
    plt.plot(df_pred, label="predictions")
    lower, upper = conf_int[:, 0], conf_int[:, 1]
    plt.fill_between(range(len(df_main)), lower, upper, color="red", alpha=0.3)
    plt.legend()
    plt.show()
    
def plot_AutoARIMA(model, df_main, df_test):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    params = model.get_params()
    d = params["order"][1]

    data_fitted = model.predict_in_sample(start=d, end=-1)
    data_pred, conf_int = model.predict(n_periods=len(df_test), return_conf_int=True)
    lower, upper = conf_int[:, 0], conf_int[:, 1]
    plt.clf()
    plt.plot(range(len(df_main)), df_main, label="org. data")
    plt.plot(range(len(data_fitted)), data_fitted, label="fitted")
    plt.plot(range(len(df_main)-len(data_pred), len(df_main)), data_pred, label="predicted")    
    plt.fill_between(range(len(df_main)-len(data_pred), len(df_main)), lower, upper, color="red", alpha=0.3)
    plt.title(f"Avg. lower: {np.mean(lower):0.2f} ---> Avg. upper: {np.mean(upper):0.2f}")
    plt.legend()
    plt.show()
    
    