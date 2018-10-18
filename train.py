import logging

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flexp import flexp
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, RobustScaler, MinMaxScaler
from tputils import csv_dump
from xgboost import XGBRegressor

# Used due to bug https://github.com/scikit-learn/scikit-learn/issues/12365
# Can be removed in sklearn=0.20.1
from _encoders import OrdinalEncoder


@click.command()
@click.option('--exp', default="exp", help='Experiment folder')
def main(exp):
    flexp.setup("./experiments", exp, with_date=True, loglevel=logging.INFO, log_filename="experiment.log.txt")

    # Load
    logging.info("Loading data")
    df_train = pd.read_csv("data/data_clean_train.csv")
    df_dev = pd.read_csv("data/data_clean_dev.csv")

    # Preprocess - split data to x and y, keep x as pd.DataFrame
    logging.info("Preprocessing")
    df_x_train, y_train = xy_split(df_train)
    df_x_dev, y_dev = xy_split(df_dev)

    # Features
    feature_transformer = FeatureTransformer()
    x_train = feature_transformer.fit_transform(df_x_train)  # Fit parameters of transformers and transform x_train
    x_dev = feature_transformer.transform(df_x_dev)  # Transformers already fitted, just transform x_dev
    feature_names = feature_transformer.get_feature_names()

    # Impute - fill missing values
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train)
    x_dev = imputer.transform(x_dev)

    # Scale - transforms columns to be around 0
    # For some methods easier training, better results, for some methods worse
    features_to_scale = ["city mpg__", "Year__", "Number of Doors__"]
    scaler = FeatureScaler(StandardScaler(), feature_names, features_to_scale)
    x_train = scaler.fit_transform(x_train)
    x_dev = scaler.transform(x_dev)

    # Fit
    logging.info("Fitting")
    model = RandomForestRegressor(n_estimators=10)
    model.fit(x_train, y_train)

    # Eval
    logging.info("Evaluating")
    y_train_pred = model.predict(x_train)
    y_dev_pred = model.predict(x_dev)

    eval_rmse(y_train, y_train_pred, y_dev, y_dev_pred)
    eval_feature_importance(model, feature_names)
    plot_histograms(x_train, feature_names)


class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        # Ugly but otherwise col_transformer.feature_names() doesn't work
        StandardScaler.get_feature_names = get_empty_feature_names
        FunctionTransformer.get_feature_names = get_empty_feature_names
        OrdinalEncoder.get_feature_names = get_empty_feature_names
        SimpleImputer.get_feature_names = get_empty_feature_names
        RobustScaler.get_feature_names = get_empty_feature_names

        # Transformer which returns the same result
        identity = FunctionTransformer(func=lambda x: x, validate=False)
        # transformer 1/x
        reciprocal = FunctionTransformer(func=lambda x: 1 / x, validate=False)

        # ColumnTransformer allows different columns or column subsets of the input
        # to be transformed separately and the results combined into a single
        # feature space.
        self.col_transformer = ColumnTransformer(
            [
                # (name, transformer, column(s))

                # ==categorical==
                # OneHotEncoder - M categories in column -> M columns
                ("Transmission Type", OneHotEncoder(), ["Transmission Type"]),
                # OrdinalEncoder - encodes categories to integer
                ("Vehicle Size", OrdinalEncoder([['Compact', 'Midsize', 'Large']]), ["Vehicle Size"]),

                # ==numerical==
                ("Number of Doors", identity, ["Number of Doors"]),  # Leave column as it is
                ("city mpg", reciprocal, ["city mpg"]),  # calculate 1/x
                ("Year", identity, ["Year"]),   # Leave column as it is
            ],
            remainder='drop'  # Drop all other remaining columns
        )

    def fit(self, X):
        self.col_transformer.fit(X)
        return self

    def transform(self, X):
        return self.col_transformer.transform(X)

    def get_feature_names(self):
        return self.col_transformer.get_feature_names()


class FeatureScaler(BaseEstimator, TransformerMixin):

    def __init__(self, scaler, feature_names, features_to_scale):
        self.scaler = scaler
        self.feature_names = feature_names
        self.features_to_scale = features_to_scale
        self.feature_ind_to_scale = [i for i, name in enumerate(feature_names) if name in features_to_scale]

        # Make sure I found indexes of all features_to_scale
        assert len(self.feature_ind_to_scale) == len(self.features_to_scale), \
            "{} {}".format(self.feature_ind_to_scale, self.features_to_scale)

    def fit(self, X):
        # Fit only selected columns, indexed by list
        self.scaler.fit(X[:, self.feature_ind_to_scale])
        return self

    def transform(self, X):
        # Scale selected columns and return scaled back to array
        X[:, self.feature_ind_to_scale] = self.scaler.transform(X[:, self.feature_ind_to_scale])
        return X


def eval_rmse(y_train, y_train_pred, y_dev, y_dev_pred):
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_dev = np.sqrt(mean_squared_error(y_dev, y_dev_pred))

    file = flexp.get_file_path("metrics.csv")
    header = ["metric", "trainset", "devset"]
    row = ["rmse", str(rmse_train), str(rmse_dev)]
    csv_dump([header, row], file)

    logging.info(", ".join(row))


def eval_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        logging.warning("Model doesn't have feature_importances_")
        return

    # Sort feature_names and feature_importances by feature_importances, decreasing order
    feature_importance = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    header = ["feature name", "feature importance"]
    file = flexp.get_file_path("feature_importance.csv")
    csv_dump([header] + feature_importance, file)


def xy_split(df):
    """
    :param pd.DataFrame df:
    :return:
    """
    feature_names = [col for col in df.columns if col != "MSRP"]  # All columns except MSRP
    df_x = df[feature_names]
    df_y = df[['MSRP']]

    # df.values extract numpy ndarray from pd.DataFrame
    # ravel() transforms 2D array to 1D array
    return df_x, df_y.values.ravel()


def plot_histograms(x_train, feature_names):
    for i, feature_name in enumerate(feature_names):
        plt.hist(x_train[:, i])
        plt.title("Histogram {}".format(feature_name))
        plt.savefig(flexp.get_file_path("histogram_{:02d}".format(i)))
        plt.clf()


# Ugly but otherwise col_transformer.feature_names() doesn't work
def get_empty_feature_names(x):
    return [""]


if __name__ == "__main__":
    main()
