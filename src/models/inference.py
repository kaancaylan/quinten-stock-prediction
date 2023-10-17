from src.data import preprocessing as pr
import datetime as dt
import numpy as np
from xgboost import XGBRegressor
from xgboost import Booster
import argparse
import pandas as pd


def inference(
        inference_start: dt.datetime, inference_end: dt.datetime, 
        prediction_start: dt.datetime, prediction_end: dt.datetime,
        model_path: str, df=None):
    model = Booster()
    model.load_model(f"src/models/{model_path}")
    if df is None:
        df = pr.get_data()
    X_test = df.loc[inference_start: inference_end]
    X_test = X_test.unstack(level=0).drop("year", axis=1)
    X_test = X_test.select_dtypes(exclude=["object", "datetime"])

    model.predict(X_test)
    y_pred = pd.Series(model.predict(X_test), X_test.index)
    top20_stocks = y_pred.sort_values()[-20:].index

    try:
        y_test = df.loc[prediction_start: prediction_end, "return"]
        y_test = y_test.groupby(level=1, group_keys=False).apply(lambda x: x.cumprod()[-1])
        y_test = y_test[y_test.index.isin(y_pred.index)]
        # error = rmse(y_test, y_pred)
        return pd.DataFrame({"y_test": y_test.loc[top20_stocks], "y_pred": y_pred.loc[top20_stocks]})
    except KeyError:
        pass

    return top20_stocks

    

def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE).

    Parameters:
    y_true (array-like): Actual (true) values.
    y_pred (array-like): Predicted values.

    Returns:
    float: The RMSE.
    """
    squared_errors = (y_true - y_pred) ** 2
    mean_squared_error = squared_errors.mean()
    rmse = np.sqrt(mean_squared_error)
    return rmse
    
    