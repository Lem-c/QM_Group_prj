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


def simple_linear_regression(df_, col_X: str, col_y: str, is_plot=True):
    if len(col_X) < 2 or len(col_y) < 1:
        print(col_X, col_y)
        raise Exception("Can't handle null column name to a dataset")

    df_X = df_[[col_X]]
    df_y = df_[col_y].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=2023)

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_prediction = model.predict(X_test)
    # Evaluating the model
    mse = mean_squared_error(y_test, y_prediction)
    r2 = r2_score(y_test, y_prediction)

    print(f'The MSE is: {mse:.3f} and R-squared is: {r2:.3f}')

    if not is_plot:
        return

    # Plot outputs with scatter and line
    plt.scatter(X_test, y_test, color="black")
    plt.plot(X_test, y_prediction, color="blue", linewidth=2)

    plt.xticks(())
    plt.yticks(())

    plt.show()
