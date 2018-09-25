import argparse
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xarray as xr

# ----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
def train_test_regression_linear(x_train,
                                 y_train,
                                 x_test,
                                 y_test):
    """
    Train and test using the linear regression model using a train/test split of single dataset.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: None
    """

    # a dictionary of model names to scores we'll populate and return
    model_scores = {}

    # create a parameter grid we'll use to parameterize the model at each iteration
    param_grid = ParameterGrid({'fit_intercept': [True, False], 'normalize': [True, False]})

    # iterate over each model parameterization
    for params in param_grid:
        model = LinearRegression(**params)
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        model_scores[score] = params

    return model_scores


# ----------------------------------------------------------------------------------------------------------------------
def train_test_regression_ridge(x_train,
                                y_train,
                                x_test,
                                y_test):
    """
    Train and test using the ridge regression model using a train/test split of single dataset.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: None
    """

    # a dictionary of model scores to parameters that we'll populate and return
    model_scores = {}

    # create a parameter grid we'll use to parameterize the model at each iteration
    alphas = [0.25, 0.5, 1.0, 2.5, 5, 10]
    max_iterations = [None, 1, 2, 5, 10, 50, 100, 1000, 100000]
    tolerances = [0.00001, 0.001, 0.01, 0.1, 1, 2, 5, 10]
    solvers = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    param_grid = ParameterGrid({'alpha': alphas, 'max_iter': max_iterations, 'tol': tolerances, "solver": solvers})

    # iterate over each model parameterization
    for params in param_grid:
        try:
            model = Ridge(**params)
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            model_scores[score] = params

        except Exception as ex:
            _logger.exception('Failed to complete', exc_info=True)
            raise

    return model_scores


# ----------------------------------------------------------------------------------------------------------------------
def train_test_regression_forest(x_train,
                                 y_train,
                                 x_test,
                                 y_test):
    """
    Train and test using the random forest regression model using a train/test split of single dataset.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: None
    """

    # a dictionary of model scores to parameters that we'll populate and return
    model_scores = {}

    # create a parameter grid we'll use to parameterize the model at each iteration
    estimators = [1, 2, 5, 10, 25, 100]
    criteria = ['mse', 'mae']
    max_features = [1, 2, 3, 0.5, 0.75, 'auto', 'sqrt', 'log2', None]
    max_depths = [None, 1, 2, 3, 5, 10, 20]
    min_samples_splits = [2, 5, 10, 0.1, 0.25, 0.5, 0.75]
    min_samples_leafs = [1, 2, 5, 10, 0.1, 0.25, 0.5, 0.75]
    bootstraps = [True, False]
    n_jobs = [-1]   # run fit jobs in parallel across all cores
    param_grid = ParameterGrid({'n_estimators': estimators,
                                'criterion': criteria,
                                'max_features': max_features,
                                'max_depth': max_depths,
                                'min_samples_split': min_samples_splits,
                                'min_samples_leaf': min_samples_leafs,
                                'bootstrap': bootstraps,
                                'n_jobs': n_jobs})

    # iterate over each model parameterization
    for params in param_grid:
        try:
            model = Ridge(**params)
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            model_scores[score] = params

        except Exception as ex:
            _logger.exception('Failed to complete', exc_info=True)
            raise

    return model_scores


# ----------------------------------------------------------------------------------------------------------------------
def train_test_regression(x_train,
                          y_train,
                          x_test,
                          y_test):
    """
    Train and test a number of regression models using a train/test split of single dataset, and log/report scores.
    Each regression model used will use its default initialization parameters.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: None
    """

    # a dictionary of model names to scores we'll populate and return
    model_scores = {}

    # create and train a linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)
    model_scores["LinearRegression"] = model.score(x_test, y_test)

    # create and train a ridge regression model
    model = Ridge()
    model.fit(x_train, y_train)
    model_scores["Ridge"] = model.score(x_test, y_test)

    # create and train a random forest regression model
    for trees in [3, 10, 20, 100, 250]:
        model = RandomForestRegressor(n_estimators=trees)
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        _logger.info("Random Forest (trees={t}) score: {result}".format(t=trees, result=score))

    # create and train a K-neighbors regression model
    for k in [1, 3, 5, 10, 20]:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        _logger.info("K-Neighbors (k={k}) score: {result}".format(k=k, result=score))

    # # create and train an Ada boost regression model, trying various estimators and learning rate parameters
    # for estimators in [1, 3, 5, 10, 20]:
    #     for rate in [0.01, 0.1, 1, 5, 12]:
    #         model = AdaBoostRegressor(n_estimators=estimators, learning_rate=rate)
    #         model.fit(x_train, y_train)
    #         score = model.score(x_test, y_test)
    #         _logger.info("Ada Boost (estimators={n}, learning rate={r}) score: {result}".format(n=estimators,
    #                                                                                             r=rate,
    #                                                                                             result=score))

    # # create and train a bagging regression model
    # model = BaggingRegressor()
    # model.fit(x_train, y_train)
    # score = model.score(x_test, y_test)
    # _logger.info("Bagging score: {result}".format(result=score))

    # create and train an extra trees regression model
    for trees in [3, 6, 10, 20]:
        model = ExtraTreesRegressor(n_estimators=trees)
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        _logger.info("Extra Trees (trees={t}) score: {result}".format(t=trees, result=score))

    # create and train a support vector regression model with an linear kernel
    model = SVR(kernel='linear', C=1e3)
    model.fit(x_train.flatten(), y_train.flatten())
    score = model.score(x_test, y_test)
    _logger.info("SVR (linear) score: {result}".format(result=score))

    # create and train a support vector regression model with a polynomial kernel
    model = SVR(kernel='poly', C=1e3, degree=2)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    _logger.info("SVR (polynomial) score: {result}".format(result=score))

    # create and train a support vector regression model with an RBF kernel
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    _logger.info("SVR (RBF) score: {result}".format(result=score))


# ----------------------------------------------------------------------------------------------------------------------
def extract_timestamps(ds,
                       initial_year,
                       initial_month,
                       initial_day):
    """

    :param ds:
    :param initial_year:
    :param initial_month:
    :param initial_day:
    :return:
    """

    # Cook up an initial datetime object based on our specified initial date.
    initial = datetime(initial_year, initial_month, initial_day)

    # Create an array of datetime objects from the time values (assumed to be in units of days since the inital date).
    times = ds.variables['time'].values
    datetimes = np.empty(shape=times.shape, dtype='datetime64[m]')
    for i in range(datetimes.size):
        datetimes[i] = initial + timedelta(days=times[i])

    # Put the array into a Series and return it.
    return pd.Series(datetimes)


# ----------------------------------------------------------------------------------------------------------------------
def pull_vars_into_dataframe(dataset,
                             variables,
                             level):
    """
    Create a pandas DataFrame from variables of an xarray DataSet.

    :param dataset:
    :param variables:
    :param level:
    :return:
    """

    # the dataframe we'll populate and return
    df = pd.DataFrame()

    # loop over each variable, adding each into the dataframe
    for var in variables:

        # if we have (time, lev, lat, lon), then use level parameter
        dimensions = dataset.variables[var].dims
        if dimensions == ('time', 'lev', 'lat', 'lon'):
            series = pd.Series(dataset.variables[var].values[:, level, :, :].flatten())
        elif dimensions == ('time', 'lat', 'lon'):
            series = pd.Series(dataset.variables[var].values[:, :, :].flatten())
        else:
            raise ValueError("Unsupported variable dimensions: {dims}".format(dims=dimensions))

        # add the series into the dataframe
        df[var] = series

    # make sure we have a generic index name
    df.index.rename('index', inplace=True)

    return df

# ----------------------------------------------------------------------------------------------------------------------
def score_models(dataset_features,
                 dataset_labels,
                 feature_vars,
                 label_vars):
    """

    :param dataset_features:
    :param dataset_labels:
    :param feature_vars:
    :param label_vars:
    :return:
    """

    # we assume each variable has multiple levels, and we loop over each
    for lev in range(dataset_features.lev.size):

        # get all features into a dataframe for this level
        df_features = pull_vars_into_dataframe(dataset_features,
                                               feature_vars,
                                               level=lev)

        # train/test for each label at this level
        for label in label_vars:

            # get the label data for this level into a dataframe
            df_labels = pull_vars_into_dataframe(dataset_labels,
                                                 [label],
                                                 level=lev)

            # split into train/test datasets
            train_x, test_x, train_y, test_y = train_test_split(df_features,
                                                                df_labels,
                                                                test_size=0.25,
                                                                random_state=4)

            # for this group of features/label perform some training/tests using various regression models
            _logger.info("Model results for features: {fs}  and label: {lbl} at level {l}".format(fs=feature_vars,
                                                                                                  lbl=label,
                                                                                                  l=lev))

            # score the linear regression model using various parameters
            score_params = train_test_regression_linear(train_x, train_y, test_x, test_y)

            best_score = np.max(np.array(list(score_params.keys())))
            best_param_set = score_params[best_score]
            print("LinearRegression")
            print("    Best parameter set: {params}".format(params=best_param_set))
            print("    Best score: {score}".format(score=best_score))

            # score the ridge regression model using various parameters
            score_params = train_test_regression_ridge(train_x, train_y, test_x, test_y)

            best_score = np.max(np.array(list(score_params.keys())))
            best_param_set = score_params[best_score]
            print("Ridge")
            print("    Best parameter set: {params}".format(params=best_param_set))
            print("    Best score: {score}".format(score=best_score))

            # score the random forest regression model using various parameters
            score_params = train_test_regression_ridge(train_x, train_y, test_x, test_y)

            best_score = np.max(np.array(list(score_params.keys())))
            best_param_set = score_params[best_score]
            print("Random Forest")
            print("    Best parameter set: {params}".format(params=best_param_set))
            print("    Best score: {score}".format(score=best_score))


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This module is used to showcase ML modeling of the climate using scikit-learn, using NCAR CAM files as input.
    """

    try:

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_flows",
                            help="NetCDF file containing flow variables",
                            nargs='*',
                            required=True)
        parser.add_argument("--input_tendencies",
                            help="NetCDF file containing time tendency forcing variables",
                            nargs='*',
                            required=True)
        args = parser.parse_args()

        # open the features (flows) and labels (tendencies) as xarray DataSets
        ds_features = xr.open_mfdataset(args.input_flows)
        ds_labels = xr.open_mfdataset(args.input_tendencies)

        # confirm that we have datasets that match on the time dimension/coordinate
        if (ds_features.variables['time'].values != ds_labels.variables['time'].values).any():
            _logger.info('ERROR: Non-matching time values')
        else:
            _logger.info("OK: time values match as expected")

        # # TODO get initial year, month, and day from the datasets
        # init_year = 2000
        # init_month = 12
        # init_day = 27
        # timestamps = extract_timestamps(ds_features, init_year, init_month, init_day)

        # # train/fit/score models using the dry features and corresponding labels
        # features = ['PS', 'T', 'U', 'V']
        # labels = ['PTTEND', 'PUTEND', 'PVTEND']
        # score_models(ds_features,
        #              ds_labels,
        #              features,
        #              labels)

        # train/fit/score models using the moist features and corresponding labels
        features = ['PRECL', 'Q']
        labels = ['SHFLX']
        score_models(ds_features,
                     ds_labels,
                     features,
                     labels)

        # # train/fit/score models using the dry and moist features and the moist labels
        # features = ['PS', 'T', 'U', 'V', 'PRECL', 'Q']
        # labels = ['SHFLX']
        # score_models(ds_features,
        #              ds_labels,
        #              features,
        #              labels)

    except Exception as ex:

        _logger.exception('Failed to complete', exc_info=True)
        raise
