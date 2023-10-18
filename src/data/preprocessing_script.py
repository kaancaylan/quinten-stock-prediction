import argparse

import numpy as np
import pandas as pd


def main(args: argparse.Namespace) -> None:
    """Preprocess raw data.

    Perform the set of preprocessing steps for the training.

    @param args: The namespace with cmd arguments.
    @return: Nothing.
    """
    # Read data.
    df = pd.read_csv(args.raw_data_path, sep=";", index_col="date")
    df = df.drop(columns="Unnamed: 0")

    # Standardize NaNs.
    df = df.replace("missing", np.NAN)

    # Cast dtypes.
    df = df.apply(pd.to_numeric, errors="ignore")

    # Drop duplicates.
    df = df.drop_duplicates()

    # Filter by the date.
    df = df[df.index > args.filter_date]

    # Filter out symbols with the small capitalisation.
    symbol_total_cap = df.groupby("symbol").marketCap.sum()
    good_symbols = symbol_total_cap[symbol_total_cap != 0].index.tolist()
    df = df[df.symbol.isin(good_symbols)]

    # Filter out features, which have no values for a specific symbol.
    all_columns = df.columns
    bad_columns = set()
    for symbol in good_symbols:
        bad_columns = bad_columns.union(
            all_columns[df[df.symbol == symbol].isna().all()]
        )

    good_columns = set(all_columns).difference(bad_columns)
    df = df[list(good_columns)]

    # Drop useless columns.
    to_drop = ["year"]
    df = df.drop(columns=to_drop)

    # Drop highly correlated features.
    correlated_columns = [
        "priceToBookRatio",
        "grahamNumber",
        "revenuePerShare",
        "grahamNetNet",
        "netIncomePerShare",
        "ptbRatio",
        "debtEquityRatio",
        "stockPrice",
        "operatingCashFlowPerShare",
        "interestDebtPerShare",
        "companyEquityMultiplier",
        "cashPerShare",
        "tangibleBookValuePerShare",
        "freeCashFlowPerShare",
        "roe",
        "debtToEquity",
        "priceBookValueRatio",
        "pfcfRatio",
        "enterpriseValue",
        "bookValuePerShare",
        "pbRatio",
        "debtRatio",
        "returnOnTangibleAssets",
    ]
    df = df.drop(columns=correlated_columns)

    # Interpolate missing feature values.
    df = df.reset_index()
    df = df.groupby("symbol").apply(lambda group: group.interpolate().ffill().bfill())
    df = df.reset_index(drop=True)

    # Dump the dataset.
    df.to_csv(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data-path",
        default="raw_data_finance.csv",
        help="Path to the file with the raw data.",
    )
    parser.add_argument(
        "--save-path",
        default="processed_data_finance.csv",
        help="Path where to save processed data.",
    )
    parser.add_argument(
        "--filter_date",
        default="2017-01-31",
        help="Records before this date will be dropped.",
    )
    main(parser.parse_args())
