from src.data import preprocessing_utils as pr
import datetime as dt
import numpy as np
from xgboost import XGBRegressor
import argparse
import pandas as pd


def inference(
        inference_start: dt.datetime, inference_end: dt.datetime,
        prediction_start: dt.datetime, prediction_end: dt.datetime,
        model_path: str=None, df=None, model=None):
    """Infer top 20 stocks based on a trained XGBoost Model.

    Args:
        inference_start (dt.datetime): Starting date for inference. This must be a date that comes later than the training period. 
                                It must also be the same duration as the training period
        inference_end (dt.datetime): Ending date period
        prediction_start (dt.datetime): Start for the predicted dates that we are inferring on. The dates must again match the duration of the
                                one from the trained model to have meaningful results.
        prediction_end (dt.datetime): End for the predicted dates that we are inferring on.
        model_path (str): The path for the model trained.
        df (pd.DataFrame, optional): If dataset is provided, the inference will be done withou loading data. 
                            If it is None, the data will be loaded by itslef Defaults to None.

    Returns:
        pd.DataFrame: the predicted returns of the top 20 stocks
    """
    if model is None:
        model = XGBRegressor()
        model.load_model(f"src/models/{model_path}")
    if df is None:
        df = pr.get_data()
    X_test = df.loc[inference_start: inference_end]
    X_test = X_test.unstack(level=0).drop("year", axis=1)
    X_test = X_test.select_dtypes(exclude=["object", "datetime"])

    model.predict(X_test)
    y_pred = pd.Series(model.predict(X_test), X_test.index)
    top20_stocks = y_pred.sort_values()[-20:]

    try:
        y_test = df.loc[prediction_start: prediction_end, "return"]
        y_test = y_test.groupby(level=1, group_keys=False).apply(lambda x: (1+x).cumprod()[-1]-1)
        y_test = y_test[y_test.index.isin(y_pred.index)]
        # error = rmse(y_test, y_pred)
        return pd.DataFrame({"y_test": y_test.loc[top20_stocks.index], "y_pred": y_pred.loc[top20_stocks.index]})
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
