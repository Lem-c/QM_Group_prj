"""
@ Author    : Lem Chen
@ Time      : 23/11/2023
"""

import AllDataFrameInOne as myDF
import util

crime_data_url = "./data/MPS Borough Level Crime (Historical).csv"
rent_data_url = "./data/local-authority-rents-borough.xlsx"
selected_crimes = ['Burglary Business and Community', 'Historical Fraud and Forgery', 'Criminal Damage', 'Bicycle Theft']

if __name__ == "__main__":
    crime_df = myDF.AllDataFrameInOne(rent_data_url, crime_data_url)

    # Simple linear
    crime_df.join_tables('Bicycle Theft', 202001, '2020-21')
    util.simple_linear_regression(crime_df.reg_df_, 'year_total', '2020-21')

    # # Multi linear
    # crime_df.join_multi_row(selected_crimes, 202001, '2020-21')
    # util.multi_linear_regression(crime_df.reg_df_, selected_crimes, '2020-21')
