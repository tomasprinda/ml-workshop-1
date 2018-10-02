import pandas as pd
import unittest

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def xy_split(df):
    """
    :param pd.DataFrame df:
    :return:
    """
    df_x = df[['Year', 'highway MPG', 'city mpg', 'Popularity']]
    df_y = df[['MSRP']]

    # df.values extract numpy ndarray from pd.DataFrame
    # ravel() transforms 2D array to 1D array
    return df_x.values, df_y.values.ravel()


class MyTest(unittest.TestCase):

    def test_packages(self):
        # Load
        df_train = pd.read_csv("/data/cardataset.zip")

        # Preprocess
        x, y = xy_split(df_train)

        # Fit
        model = RandomForestRegressor(n_estimators=10)
        model.fit(x, y)

        # Eval
        y_pred = model.predict(x)
        mse = mean_squared_error(y, y_pred)
        self.assertLess(mse, 1e10)
