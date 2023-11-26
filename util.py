# This is assist methods file
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
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
    print(f'The coefficients is(are): {coefficients} and intercept is: {intercept}')
    return y_prediction


def plot_reg(X_test, y_test, y_prediction):
    # Plot outputs with scatter and line
    plt.scatter(X_test, y_test, color="black")
    plt.plot(X_test, y_prediction, color="blue", linewidth=2)

    plt.xticks(())
    plt.yticks(())

    plt.show()


def new_plot(X_test, y_test, y_prediction, R_square):
    plt.style.use('default')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(X_test, y_prediction, color='k', label='Regression model')
    ax.scatter(X_test, y_test, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
    ax.set_ylabel('Gas production (Mcf/day)', fontsize=14)
    ax.set_xlabel('Porosity (%)', fontsize=14)
    ax.text(0.8, 0.1, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
            transform=ax.transAxes, color='grey', alpha=0.5)
    ax.legend(facecolor='white', fontsize=11)
    ax.set_title('$R^2= %.2f$' % R_square, fontsize=18)

    fig.tight_layout()


def simple_linear_regression(df_, col_X: str, col_y: str, is_plot=True):
    if len(col_X) < 2 or len(col_y) < 1:
        print(col_X, col_y)
        raise Exception("Can't handle null column name to a dataset")

    df_X = df_[[col_X]]
    df_y = df_[col_y].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, train_size=0.7, random_state=2023)

    y_prediction = linear_predict_model(X_train, X_test, y_train, y_test)

    if is_plot:
        plot_reg(X_test, y_test, y_prediction)


def multi_linear_regression(df, crime_list: list, col_y: str):
    cleaned_data = df.dropna()

    # Defining the independent X and dependent y variables
    X = cleaned_data[crime_list]
    y = cleaned_data[col_y]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9, random_state=42)

    y_prediction = linear_predict_model(X_train, X_test, y_train, y_test)
