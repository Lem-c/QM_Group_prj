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
    print(crime_df.borough_sum_df_)
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
    combined_df = pd.DataFrame({"crime": crime_x, "rent": rent_y})

    rows_to_remove = []
    # clean data
    for index, row in combined_df.iterrows():
        current_rent = float(row['rent'])
        if current_rent < 1:
            rows_to_remove.append(index)
    combined_df.drop(rows_to_remove, inplace=True)

    # cols_to_remove = set()
    # # clean data
    # for index, col in rent.iterrows():
    #     for col_name, col_val in col.items():
    #         print(col_val)
    #         if float(col_val) - 1 < 1:
    #             cols_to_remove.add(col_name)
    #
    # rent.drop(columns=cols_to_remove, inplace=True)
    #
    # util.plot_crime_trend(rent)

    util.plot_scatter(combined_df, 'crime', 'rent')
    util.simple_linear_regression(combined_df, "crime", "rent", isPolynomial=False, polynomialDegree=1)

    # Remove the unnamed index column
    data_cleaned = combined_df.iloc[:, 0:2]
    print(data_cleaned)

    # Check for missing values
    missing_values = data_cleaned.isnull().sum()

    # Check for duplicate entries
    duplicate_entries = data_cleaned.duplicated().sum()

    print(missing_values, duplicate_entries)

    import matplotlib.pyplot as plt
    import scipy.stats as stats


# ---Create QQ plots for both 'crime' and 'rent' columns
    plt.figure(figsize=(12, 6))

    # QQ plot for 'crime'
    plt.subplot(1, 2, 1)
    stats.probplot(data_cleaned['crime'], dist="norm", plot=plt)
    plt.title('QQ Plot for Crime Data')

    # QQ plot for 'rent'
    plt.subplot(1, 2, 2)
    stats.probplot(data_cleaned['rent'], dist="norm", plot=plt)
    plt.title('QQ Plot for Rent Data')

    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------
    import statsmodels.api as sm

    # Define the dependent and independent variables
    X = data_cleaned['crime']  # Independent variable
    y = data_cleaned['rent']  # Dependent variable

    # Add a constant to the independent variable (for the intercept term)
    X_with_constant = sm.add_constant(X)

    # Fit a linear regression model
    model = sm.OLS(y, X_with_constant).fit()

    # Summary of the regression model
    model_summary = model.summary()
    print(model_summary)

# -----------------------------------------------------------------------
    from scipy import stats

    # Extract the residuals
    residuals = model.resid

    # Normality test on the residuals (using the Jarque-Bera test)
    jb_test = stats.jarque_bera(residuals)

    # Homoscedasticity test (using the Breusch-Pagan test)
    bp_test = sm.stats.diagnostic.het_breuschpagan(residuals, X_with_constant)

    # Plotting residuals to visually inspect for homoscedasticity
    plt.figure(figsize=(10, 6))
    plt.scatter(X, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Crime Rate')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Crime Rate')
    plt.show()

    print(jb_test)
    print(bp_test)

