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
    plt.fill_between(
        df_test.index,
        lower, upper,
        color="red",
        alpha=0.3
    )
    plt.title(f"Avg. lower: {np.mean(lower):0.2f} ---> Avg. upper: {np.mean(upper):0.2f}")
    plt.legend()
    plt.show()