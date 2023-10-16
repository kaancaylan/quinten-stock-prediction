import pandas as pd
import numpy as np


def get_data(path="raw_data_finance.csv"):
    date_cols = ["date", "acceptedDate", "fillingDate"]
    df = pd.read_csv(path, delimiter=";", parse_dates=date_cols).iloc[:, 1:]
    df = df.replace("missing", np.nan)

    # set index on symbol datetime
    df.set_index(["date", "symbol"], inplace=True)

    # Drop duplicates on index
    df = df[~df.index.duplicated(keep='first')]

    # convert all values into numeric
    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # Outlier columns that cannot be auto handles
    irrelevant_cols = ["cik", "finalLink", "link"]

    df["period"] = df["period"].replace("FY", "Q4").apply(lambda x:int(x[-1]) if isinstance(x, str) else x)

    # Analyst Recommendations handling

    recommendation_mapping = {
        "Strong Buy": 2,
        "Buy": 1,
        "Neutral": 0,
        "Sell": -1,
        "Strong Sell": -2
    }
    recommendation_cols = df.filter(regex="Recommendation").columns
    for col in recommendation_cols:

        df[col] = df[col].map(recommendation_mapping)
    df = add_return(df)

    return df.drop(columns=irrelevant_cols).sort_index(level=0)


def add_return(df):
    df["return"] = df["close"].groupby(level="symbol", group_keys=False).apply(lambda x: x.pct_change())
    return df
