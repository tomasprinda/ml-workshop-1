import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score


def main():
    # Load
    df_train = pd.read_csv("data/data_clean_train.csv")
    df_dev = pd.read_csv("data/data_clean_dev.csv")

    # Preprocess
    x_train, y_train = xy_split(df_train)
    x_dev, y_dev = xy_split(df_dev)

    # Fit
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Eval
    y_dev_pred = model.predict(x_dev)
    acc = accuracy_score(y_dev, y_dev_pred)
    print("acc={}".format(acc))


def xy_split(df):
    """
    :param pd.DataFrame df:
    :return:
    """
    df_x = df[['Year', 'highway MPG', 'city mpg', 'Popularity']]
    df_y = df[['MSRP']] > 30000

    # df.values extract numpy ndarray from pd.DataFrame
    # ravel() transforms 2D array to 1D array
    return df_x.values, df_y.values.ravel()


if __name__ == "__main__":
    main()
