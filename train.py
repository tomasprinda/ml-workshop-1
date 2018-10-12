import logging

import click
import pandas as pd
import numpy as np
from flexp import flexp
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from tputils import csv_dump


@click.command()
@click.option('--exp', default="exp", help='Experiment folder')
def main(exp):
    flexp.setup("./experiments", exp, with_date=True, log_filename="experiment.log.txt")

    # Load
    logging.debug("Loading data")
    df_train = pd.read_csv("data/data_clean_train.csv")
    df_dev = pd.read_csv("data/data_clean_dev.csv")

    # Preprocess
    logging.debug("Preprocessing")
    x_train, y_train = xy_split(df_train)
    x_dev, y_dev = xy_split(df_dev)

    # Fit
    logging.debug("Fitting")
    model = RandomForestRegressor(n_estimators=10)
    model.fit(x_train, y_train)

    # Eval
    logging.debug("Evaluating")
    y_train_pred = model.predict(x_train)
    y_dev_pred = model.predict(x_dev)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_dev = np.sqrt(mean_squared_error(y_dev, y_dev_pred))

    rmse_file = flexp.get_file_path("metrics.csv")
    header = ["metric", "trainset", "devset"]
    row = ["rmse", str(rmse_train), str(rmse_dev)]
    csv_dump([header, row], rmse_file)

    logging.debug(", ".join(row))


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


if __name__ == "__main__":
    main()
