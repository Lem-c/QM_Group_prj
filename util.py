# This is assist methods file
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def format_read_csv(file_: str, sheet_id=1, header_id=0) -> pd.DataFrame:
    file_type = os.path.splitext(file_)[1]
    table_df = None

    if file_type == ".xlsx" or file_type == ".xls":
        table_df = pd.read_excel(file_, sheet_name=sheet_id, header=header_id)
    else:
        table_df = pd.read_csv(file_)

    if table_df is None:
        raise FileNotFoundError("Read in xls file fail")

    return table_df


def linear_predict_model(X_train, X_test, y_train, y_test):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_prediction = model.predict(X_test)
    # Evaluating the model
    mse = mean_squared_error(y_test, y_prediction)
    r2 = r2_score(y_test, y_prediction)
    # Model coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_

    print(f'The MSE is: {mse:.3f} and R-squared is: {r2:.3f}')
    print(f'The coefficient(s) is(are): {coefficients} and intercept is: {intercept}')
    return y_prediction


def plot_reg(X_test, y_test, y_prediction):
    # Plot outputs with scatter and line
    plt.scatter(X_test, y_test, s=10, label="Original Data", color="black")
    plt.plot(X_test, y_prediction, color="m")
    plt.xlabel("Independent variable (X)")
    plt.ylabel("Dependent variable (Y)")

    plt.show()


def simple_linear_regression(df_, col_X: str, col_y: str, isPolynomial=False, polynomialDegree=2, is_plot=True):
    if len(col_X) < 2 or len(col_y) < 1:
        print(col_X, col_y)
        raise Exception("Can't handle null column name to a dataset")

    df_X = df_[[col_X]].to_numpy()
    df_y = df_[col_y].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, train_size=0.7, random_state=2023)

    if isPolynomial:
        # Reshaping data for the model
        df_y = df_y[:, np.newaxis]
        # Transforming the data to include another axis
        polynomial_features = PolynomialFeatures(degree=polynomialDegree)
        X_poly = polynomial_features.fit_transform(df_X)
        # Not split the dataset
        X_train = X_poly
        X_test = X_poly
        y_train = df_y
        y_test = df_y

    y_prediction = linear_predict_model(X_train, X_test, y_train, y_test)

    if is_plot and not isPolynomial:
        plot_reg(X_test, y_test, y_prediction)
    else:
        # Combine X and Y into a single array for sorting
        combined = np.column_stack((df_X, y_prediction))
        # Sort the array by the first column (X)
        sorted_combined = combined[np.argsort(combined[:, 0])]
        # Extract the sorted X and Y values
        X_sorted = sorted_combined[:, 0]
        Y_sorted = sorted_combined[:, 1]
        plot_reg(X_sorted, df_y, Y_sorted)


def multi_linear_regression(df, crime_list: list, col_y: str):
    print("----Multi-Linear-Regression----")
    cleaned_data = df.dropna()

    # Defining the independent X and dependent y variables
    X = cleaned_data[crime_list].to_numpy()
    y = cleaned_data[col_y].to_numpy()

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42)

    linear_predict_model(X_train, X_test, y_train, y_test)


def sum_table_by_year(df_, col_year="YearMonth"):
    """
    Can only be used after pivoted by year
    """

    # Grouping the data by 'LookUp_BoroughName' and summing up all the monthly crime counts
    df_ = df_.groupby([col_year]).sum()
    # Resetting the index to have 'MajorText' as a column
    df_.reset_index(inplace=True)
    return df_


def plot_crime_trend(df_, col_year='YearMonth'):
    # Extracting YearMonth for the x-axis
    x = df_[col_year]
    print(df_.columns)

    # Plotting line graphs for each borough
    plt.figure(figsize=(15, 6))
    for column in df_.columns[1:]:  # Skipping the first few columns to get only boroughs
        plt.plot(x, df_[column], label=column)

    # Adding labels and title
    plt.xlabel(col_year)
    plt.ylabel('Values')
    plt.title('Total number of crimes by Borough Over Time')
    plt.xticks(rotation=45)  # Rotating x-axis labels for better readability
    plt.legend()
    plt.show()


def polynomial_regression(df, ):
    return
