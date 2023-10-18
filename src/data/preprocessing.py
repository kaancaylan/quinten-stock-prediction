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
    df = feature_engineering(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)


    return df.drop(columns=irrelevant_cols).sort_index(level=0)


def add_return(df):
    df["return"] = df["close"].groupby(level="symbol", group_keys=False).apply(lambda x: x.pct_change())
    return df


def calculate_eps(net_income, weighted_average_shs_out):
    # Use the .replace method to replace 0 with NaN to avoid division by zero
    weighted_average_shs_out_replaced = weighted_average_shs_out.replace(
        0, np.nan)
    # Now perform element-wise division
    eps = net_income / weighted_average_shs_out_replaced
    return eps


def feature_engineering(df):
    df['ROI'] = (df['capitalExpenditure']/df['netIncome']) * 100
    df['revenue'] = df['revenue'].replace(0, np.nan)
    df['profit_margin'] = (df['netIncome'] / df['revenue']) * 100
    df['weightedAverageShsOut'] = df['weightedAverageShsOut'].replace(
        0, np.nan)
    eps = calculate_eps(df['netIncome'], df['weightedAverageShsOut'])
    df['eps'] = eps
    df['eps'] = df['eps'].replace(0, np.nan)
    df['market_value_per_share'] = df['marketCapitalization'] / \
        df['weightedAverageShsOut']
    df['pe_ratio'] = df['market_value_per_share'] / df['eps']

    return df
