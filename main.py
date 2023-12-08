"""
@ Author    : Lem Chen
@ Time      : 23/11/2023
"""
import pandas as pd

import AllDataFrameInOne as myDF
import util
import statsmodels.formula.api as smf
import statsmodels

crime_data_url = "./data/MPS Borough Level Crime (Historical).csv"
rent_data_url = "./data/local-authority-rents-borough.xlsx"
selected_minor_crimes = ['Theft from Person', 'Rape', 'Historical Fraud and Forgery']
selected_major_crimes = ['Robbery', 'Possession of Weapons', 'Theft']

if __name__ == "__main__":
    crime_df = myDF.AllDataFrameInOne(rent_data_url, crime_data_url)

    crime_df.print_column_names()
    # plot the total crime trend figure
    crime_df.join_all_text(["MajorText", "MinorText", "LookUp_BoroughName"],
                           "LookUp_BoroughName",
                           "CrimeCount", side='left')
    crime_df.join_all_together(is_change=True)

    borough = crime_df.borough_sum_df_

    crime_df.join_all_text(["Code", "New Code", "Area"],
                           "Area",
                           "RentCount", side='right')

    rent = crime_df.borough_sum_df_

    # Convert 'YearMonth' in both dataframes to string (object) type
    borough['YearMonth'] = borough['YearMonth'].astype(str)
    rent['YearMonth'] = rent['YearMonth'].astype(str)
    new_join = pd.merge(borough, rent,
                        on='YearMonth',
                        how='left',
                        suffixes=('_crime', '_rent'))

    # Step 1: Identify columns with _crime and _rent suffixes
    crime_cols = [col for col in new_join.columns if col.endswith('_crime')]
    rent_cols = [col for col in new_join.columns if col.endswith('_rent')]

    # Step 2: Melt the DataFrame for each set of columns
    crime_melted = new_join.melt(id_vars=[col for col in new_join.columns if col not in crime_cols],
                           value_vars=crime_cols,
                           var_name='crime_category',
                           value_name='crime')

    rent_melted = new_join.melt(id_vars=[col for col in new_join.columns if col not in rent_cols],
                          value_vars=rent_cols,
                          var_name='rent_category',
                          value_name='rent')

    crime_x = crime_melted['crime']
    rent_y = rent_melted['rent']

    # Combine the melted DataFrames
    # This step depends on how you want to combine them. If they share a common identifier, you can merge them
    # For example, if they share an 'id' column, you would merge like this:
    combined_df = pd.DataFrame({"crime":crime_x, "rent":rent_y})
    print(combined_df)

    util.plot_scatter(combined_df, 'crime', 'rent')

    # # plot all borough line graph
    # util.plot_crime_trend(crime_df.borough_sum_df_)
    # # plot a borough cases in each month
    # util.plot_bar_chart(crime_df.borough_sum_df_, "Barnet")
    #
    # # Simple linear
    # crime_df.join_minor_tables('Burglary', 202001, ['2020-21'], minor_major='MajorText')
    #

    # reg_wb_green = smf.ols(formula='crime ~ rent', data=combined_df).fit()
    # print(reg_wb_green.summary())
    #
    # util.plot_scatter(crime_df.reg_df_, 'year_total', '2020-21')
    # util.simple_linear_regression(crime_df.reg_df_, 'year_total', '2020-21', isPolynomial=False, polynomialDegree=1)
    util.simple_linear_regression(combined_df, 'crime', 'rent', isPolynomial=False, polynomialDegree=1)
    #
    # # Multi linear
    # crime_df.join_multi_row(selected_major_crimes, 202001, '2020-21', text_column='MajorText')
    # # crime_df.print_column_names()
    # util.multi_linear_regression(crime_df.reg_df_, selected_major_crimes, '2020-21')
    # util.stats_model(crime_df.reg_df_, selected_major_crimes, '2020-21')
