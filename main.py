"""
@ Author    : Lem Chen
@ Time      : 23/11/2023
"""

import AllDataFrameInOne as myDF
import util

crime_data_url = "./data/MPS Borough Level Crime (Historical).csv"
rent_data_url = "./data/local-authority-rents-borough.xlsx"
selected_minor_crimes = ['Theft from Person', 'Rape', 'Historical Fraud and Forgery']
selected_major_crimes = ['Robbery', 'Possession of Weapons', 'Theft', 'Sexual Offences']

if __name__ == "__main__":
    crime_df = myDF.AllDataFrameInOne(rent_data_url, crime_data_url)
    # plot the total crime trend figure
    crime_df.join_all_text(["MajorText", "MinorText", "LookUp_BoroughName"],
                           "LookUp_BoroughName",
                           "CrimeCount")
    # plot all borough line graph
    util.plot_crime_trend(crime_df.borough_sum_df_)
    # plot a borough cases in each month
    util.plot_bar_chart(crime_df.borough_sum_df_, "Barnet")

    # Simple linear
    crime_df.join_minor_tables('Possession of Firearms Offences', 202001, '2020-21')
    util.simple_linear_regression(crime_df.reg_df_, 'year_total', '2020-21', isPolynomial=True, polynomialDegree=1)

    # Multi linear
    crime_df.join_multi_row(selected_major_crimes, 202001, '2020-21', text_column='MajorText')
    # crime_df.print_column_names()
    util.multi_linear_regression(crime_df.reg_df_, selected_major_crimes, '2020-21')
