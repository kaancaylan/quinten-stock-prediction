import src.data.preprocessing as pr
import datetime as dt
import numpy as np
from xgboost import XGBRegressor
import argparse


def train(
        training_start: dt.datetime, training_end: dt.datetime,
        validation_start: dt.datetime, validation_end: dt.datetime,
        model_name: str = "Model1"
):
    df = pr.get_data()
    X_train = df.loc[training_start: training_end]
    X_train = X_train.unstack(level=0).drop("year", axis=1)

    # Testing data
    test_per_rets = df.loc[validation_start: validation_end, "return"]

    X_train, y_train = modeling_prep(X_train, test_per_rets)

    model = XGBRegressor()
    model.fit(X_train.to_numpy(), y_train)
    model.save_model("models/{model_name}")
    




def modeling_prep(X_train, test_per_rets):
    # drop stocks with missing returns for this period
    to_drop = test_per_rets[test_per_rets.isna()].index.get_level_values("symbol").unique()

    # drop stocks that traded under 1$ in the period
    under_1 = X_train['close'].groupby(level="symbol").apply(lambda x: (x<1).any(axis=1)).droplevel(1)
    to_drop = to_drop.union(under_1[under_1].index).unique()

    X_train = X_train.drop(to_drop)
    y_train = test_per_rets.drop(to_drop, level=1)
    
    assert len(np.setdiff1d(X_train.index, y_train.index.get_level_values(1))) == 1
    assert len(np.setdiff1d(y_train.index.get_level_values(1), X_train.index)) == 1

    y_train = y_train.groupby(level=1, group_keys=False).apply(lambda x: x.cumprod()[-1])

    assert all(X_train.index == y_train.index)

    X_train = X_train.select_dtypes(exclude=["object", "datetime"])

    return X_train, y_train


def main():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("training_start", 
    help="Date for the start of training. Format should be YYY/MM/DD, the day should be the last day of month. ", type=str)
    parser.add_argument("training_end", 
    help="Date for the end of training. Format should be YYY/MM/DD, the day should be the last day of month. ", type=str)

if __name__ == "__main__":
    main()