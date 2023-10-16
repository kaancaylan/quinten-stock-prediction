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


    irrelevant_cols = ["finalLink", "link", "reportedCurrency"]
    df["period"] = df["period"].replace("FY", "Q4").apply(lambda x:int(x[-1]) if isinstance(x, str) else x)
    df["rating"] = pd.Categorical(df["rating"])

    # Analyst Recommendations
    recommendation_cols = df.filter(regex="Recommendation").columns
    for col in recommendation_cols:
        df[col] = pd.Categorical(df[col])

    return df.drop(columns=irrelevant_cols)

    
