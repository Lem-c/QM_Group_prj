"""
@ Author    : Lem Chen
@ Time      : 23/11/2023
"""

import AllDataFrameInOne as myDF
import util

crime_data_url = "./data/MPS Borough Level Crime (Historical).csv"
rent_data_url = "./data/local-authority-rents-borough.xlsx"

if __name__ == "__main__":
    crime_df = myDF.AllDataFrameInOne(rent_data_url, crime_data_url)
    crime_df.join_tables('Criminal Damage', 202001, '2020-21')
    print(crime_df.df_reg_.head())

    util.simple_linear_regression(crime_df.df_reg_, 'year_total', '2020-21')
