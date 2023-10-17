from src.data import preprocessing as pr
import datetime as dt
import numpy as np
from xgboost import XGBRegressor
import argparse



def train(
        training_start: dt.datetime, training_end: dt.datetime,
        validation_start: dt.datetime, validation_end: dt.datetime,
        model_name: str = "Model1", df=None
):
    """Train the model.
        All the date values must exactly match the ones in the dataset.

    Args:
        training_start (dt.datetime): The date in which training period will start
        training_end (dt.datetime): The date in which the training period will end
        validation_start (dt.datetime): The date in which the validation period starts. Validation period is used 
                                    as the label for the training data. The period is usually taken as 6 months.
        validation_end (dt.datetime): The date in which the validation period will end.
        model_name (str, optional): _description_. Defaults to "Model1".
    """
    if df is None:
        df = pr.get_data()
    X_train = df.loc[training_start: training_end]
    X_train = X_train.unstack(level=0).drop("year", axis=1)

    # Testing data
    test_per_rets = df.loc[validation_start: validation_end, "return"]

    X_train, y_train = modeling_prep(X_train, test_per_rets)

    model = XGBRegressor()
    model.fit(X_train.to_numpy(), y_train)
    model.save_model(f"src/models/{model_name}")
    return  model
    




def modeling_prep(X_train, test_per_rets):
    # drop stocks with missing returns for this period
    to_drop = test_per_rets[test_per_rets.isna()].index.get_level_values("symbol").unique()

    # drop stocks that traded under 1$ in the period
    under_1 = X_train['close'].groupby(level="symbol").apply(lambda x: (x<1).any(axis=1)).droplevel(1)
    to_drop = to_drop.union(under_1[under_1].index).unique()

    X_train = X_train.drop(to_drop)
    y_train = test_per_rets.drop(to_drop, level=1)
    y_train = y_train.groupby(level=1, group_keys=False).apply(lambda x: x.cumprod()[-1])
    
   
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
    help="Date for the start of training. Format should be YYY/MM/DD, the day should be the last day of month. ", type=str)
    parser.add_argument("--training_end", 
    help="Date for the end of training. Format should be YYY/MM/DD, the day should be the last day of month. ", type=str)
    parser.add_argument("--validation_start", help="Date for the start of validation period", type=str)
    parser.add_argument("--validation_end", help="Date for the start of validation period", type=str)
    parser.add_argument("--model_name", type=str)

    args = parser.parse_args()
    main(args)
