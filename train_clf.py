import itertools
import logging

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flexp import flexp
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, \
    confusion_matrix
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from tputils import csv_dump

from train import FeatureScaler, FeatureTransformer


@click.command()
@click.option('--exp', default="exp", help='Experiment folder')
def main(exp):
    flexp.setup("./experiments", exp, with_date=False, loglevel=logging.INFO, override_dir=True)

    # Load
    logging.info("Loading data")
    df_train = pd.read_csv("data/data_clean_train.csv")
    df_dev = pd.read_csv("data/data_clean_dev.csv")

    # Preprocess - split data to x and y, keep x as pd.DataFrame
    logging.info("Preprocessing")
    df_x_train, y_train = xy_split(df_train)
    df_x_dev, y_dev = xy_split(df_dev)

    # Transform labels
    # We want a classification task so we create 3 classes based on price
    # encode='ordinal' means we want output to be 0, 1, 2, other option is onehot
    # strategy="quantile" means we want in each class the same number of examples
    # other options are "uniform" or "kmeans"
    enc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy="quantile")
    # ravel() transforms to 1D and astype to integer
    y_train = enc.fit_transform(y_train).ravel().astype(np.int)
    y_dev = enc.transform(y_dev).ravel().astype(np.int)

    # converts the class ids to readable format - string
    class_names = ["cheap", "middle class", "expensive"]
    y_train = transform_labels(y_train, class_names)
    y_dev = transform_labels(y_dev, class_names)

    # Features
    feature_transformer = FeatureTransformer()
    # Fit parameters of transformers and transform x_train
    x_train = feature_transformer.fit_transform(df_x_train)
    # Transformers already fitted, just transform x_dev
    x_dev = feature_transformer.transform(df_x_dev)
    # x_train, x_dev is np.array now, we still want to know
    # the names of features
    feature_names = feature_transformer.get_feature_names()

    # Impute - fill missing values with median of the column
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train)
    x_dev = imputer.transform(x_dev)

    # Scale - transforms columns to be around 0
    # For some methods easier training, better results, for some methods worse
    features_to_scale = ["city mpg trans__", "Year__", "Number of Doors__", "Engine HP__"]
    scaler = FeatureScaler(StandardScaler(), feature_names, features_to_scale)
    x_train = scaler.fit_transform(x_train)
    x_dev = scaler.transform(x_dev)

    # Logging
    # It is useful to log almost everything for easy debugging
    logging.info("x_train.shape={} y_train.shape={}".format(x_train.shape, y_train.shape))
    logging.info("x_dev.shape={} y_dev.shape={}".format(x_dev.shape, y_dev.shape))

    # Fit
    logging.info("Fitting")
    model = RandomForestClassifier(n_estimators=10)
    model.fit(x_train, y_train)

    # Eval
    logging.info("Evaluating")
    y_train_pred = model.predict(x_train)
    y_dev_pred = model.predict(x_dev)

    eval_metrics(y_train, y_train_pred, y_dev, y_dev_pred)
    eval_confusion_matrix(y_train, y_train_pred, y_dev, y_dev_pred, class_names)


def eval_metrics(y_train, y_train_pred, y_dev, y_dev_pred):

    # Calculates accuracy
    # With accuracy there is a problem with imbalanced classes
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_dev = accuracy_score(y_dev, y_dev_pred)

    # Calculates other metrics:
    #
    # For binary classification use:
    # average='binary', pos_label='expensive'
    #
    # For multiclass classification use:
    # average='weighted', omit pos_label
    #
    precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(
        y_train, y_train_pred,  average='weighted'
    )
    precision_dev, recall_dev, f1_dev, _ = precision_recall_fscore_support(
        y_dev, y_dev_pred,  average='weighted'
    )

    # Print the metrics to csv file
    file = flexp.get_file_path("metrics.csv")
    rows = [
        ["metric", "trainset", "devset"],
        ["accuracy", str(acc_train), str(acc_dev)],
        ["precision", str(precision_train), str(precision_dev)],
        ["recall", str(recall_train), str(recall_dev)],
        ["f1", str(f1_train), str(f1_dev)],
    ]
    csv_dump(rows, file)

    logging.info(rows)


def eval_confusion_matrix(y_train, y_train_pred, y_dev, y_dev_pred, classes):

    # Calculates confusion matrix. classes used to get correct ordering
    cm = confusion_matrix(y_dev, y_dev_pred, classes)

    # Plot calculated confusion matrix. Don't normalize
    plot_confusion_matrix(cm, classes, normalize=False, filename=flexp.get_file_path("confusion_matrix_dev.png"))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,
                          filename="confusion_matrix.png"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logging.info("Normalized confusion matrix")
    else:
        logging.info('Confusion matrix, without normalization')

    logging.info(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.savefig(filename)
    plt.clf()


def transform_labels(y, class_names):
    """
    Takes integers y[i] and transforms them to class_names[y[i]]
    :param np.ndarray y:
    :param list[str] class_names:
    :return np.ndarray:
    """
    return np.array([class_names[x] for x in y])


def xy_split(df):
    """
    :param pd.DataFrame df:
    :return:
    """
    feature_names = [col for col in df.columns if col != "MSRP"]  # All columns except MSRP
    df_x = df[feature_names]

    df_y = df[['MSRP']]

    # df.values extract numpy ndarray from pd.DataFrame
    return df_x, df_y.values


if __name__ == "__main__":
    main()
