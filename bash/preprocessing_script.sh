#!/bin/bash
# This script will run the data preprocessing step and dump the resulted dataset
# to the desired directory.

python src/data/preprocessing_script.py --raw-data-path data/raw/raw_data_finance.csv --save-path data/interim/stocks_drop_n_interpolated.csv
