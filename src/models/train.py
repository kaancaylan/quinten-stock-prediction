import argparse
import datetime as dt
import numpy as np
from xgboost import XGBRegressor
import argparse
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import TimeSeriesSplit




def train(
        training_start: dt.datetime, training_end: dt.datetime,
        validation_start: dt.datetime, validation_end: dt.datetime,
        model_name: str = "Model1", df=None, hyperparam_tuning=False,
):
    """Train the model based on a given set of training and validation dates.
        All the date values must exactly match the ones in the dataset.

    Args:
        training_start (dt.datetime): The date in which training period will start
        training_end (dt.datetime): The date in which the training period will end
        validation_start (dt.datetime): The date in which the validation period starts. Validation period is used 
                                    as the label for the training data. The period is usually taken as 6 months.
        validation_end (dt.datetime): The date in which the validation period will end.
        model_name (str, optional): the name for the model for saving purposes. Defaults to "Model1".
        df (pd.DataFrame, optional): Main dataset for training the data, the data will be loaded automatically if no argument is provided.
    """
    if df is None:
        df = pr.get_data()
    X_train = df.loc[training_start: training_end]
    if len(X_train)==0:
        return None
    X_train = X_train.unstack(level=0).drop("year", axis=1)
    X_train = X_train.select_dtypes(exclude=["object", "datetime"])

    # Testing data
    test_per_rets = df.loc[validation_start: validation_end, "return"]

    X_train, y_train = modeling_prep(X_train, test_per_rets)
    model = XGBRegressor()

    if hyperparam_tuning:
        tscv = TimeSeriesSplit(n_splits=3)
        param_grid = {
            'max_depth': [3, 5, 7],
            'reg_lambda': [0.1, 1, 2],
            'n_estimators': [50, 100, 150],
            'reg_alpha': [0.1, 1 ,2]
            }
        grid_search = HalvingGridSearchCV(model, param_grid, cv=tscv,  n_jobs=4)
        return grid_search.best_estimator_

    model.fit(X_train.to_numpy(), y_train)
    model.save_model(f"src/models/{model_name}")
    return model
    

def modeling_prep(X_train, test_per_rets):
    # drop stocks with missing returns for this period
    to_drop = test_per_rets[test_per_rets.isna()].index.get_level_values("symbol").unique()

    # drop stocks that traded under 1$ in the period
    under_1 = X_train['close'].groupby(level="symbol").apply(lambda x: (x < 1).any(axis=1)).droplevel(1)
    to_drop = to_drop.union(under_1[under_1].index).unique()
    to_drop = to_drop[to_drop.isin(X_train.index)]

    X_train = X_train.drop(to_drop)
    y_train = test_per_rets.drop(to_drop, level=1)

    drop_from_x = np.setdiff1d(X_train.index, y_train.index.get_level_values(1))
    drop_from_y = np.setdiff1d(y_train.index.get_level_values(1), X_train.index)
    X_train = X_train.drop(drop_from_x)
    y_train = y_train.drop(drop_from_y, level=1)
    y_train = y_train.groupby(level=1, group_keys=False).apply(lambda x: (1+x).cumprod()[-1]-1)
    
   
    assert all(X_train.index == y_train.index)

    X_train = X_train.select_dtypes(exclude=["object", "datetime"])

    return X_train, y_train


def main(args):
    training_start = dt.datetime.strptime(args.training_start, "%Y/%m/%d")
    training_end = dt.datetime.strptime(args.training_end, "%Y/%m/%d")
    val_start = dt.datetime.strptime(args.validation_start, "%Y/%m/%d")
    val_end = dt.datetime.strptime(args.validation_end, "%Y/%m/%d")

    train(training_start, training_end, val_start, val_end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--training_start",
                        help="Date for the start of training. Format should be YYY/MM/DD, the day should be the last "
                             "day of month. ",
                        type=str)
    parser.add_argument("--training_end",
                        help="Date for the end of training. Format should be YYY/MM/DD, the day should be the last "
                             "day of month. ",
                        type=str)
    parser.add_argument("--validation_start", help="Date for the start of validation period", type=str)
    parser.add_argument("--validation_end", help="Date for the start of validation period", type=str)
    parser.add_argument("--model_name", type=str)

    args = parser.parse_args()
    main(args)
