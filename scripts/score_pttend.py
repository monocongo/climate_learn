import argparse
from datetime import datetime, timedelta
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsRegressor
import xarray as xr

# ----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
def score_regression_kneighbors(x_train,
                                y_train,
                                x_test,
                                y_test):
    """
    Train and test using the linear regression model using a train/test split of single dataset.
    Returns the best score and corresponding parameter list.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: best score (float) and corresponding list of parameters
    """

    # the best score result for a fit and its corresponding a parameter set
    best_score = sys.float_info.min
    best_params = None

    # create a parameter grid we'll use to parameterize the model at each iteration
    neighbors = [50, 100, 200]
    weights = ['uniform', 'distance']
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_sizes = [10, 30, 50]
    powers = [1, 2]
    jobs = [-1]   # -1 uses all processors
    param_grid = ParameterGrid({'n_neighbors': neighbors,
                                'weights': weights,
                                'algorithm': algorithms,
                                'leaf_size': leaf_sizes,
                                'p': powers,
                                'n_jobs': jobs})

    # iterate over each model parameterization
    for params in param_grid:
        _logger.info("\t\tFitting/scoring K-Neighbors with parameters: {p}".format(p=params))
        model = KNeighborsRegressor(**params)
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        _logger.info("\t\t\tScore: {s}".format(s=score))
        if score > best_score:
            best_score = score
            best_params = params

    return best_score, best_params


# ----------------------------------------------------------------------------------------------------------------------
def score_regression_linear(x_train,
                            y_train,
                            x_test,
                            y_test):
    """
    Train and test using the linear regression model using a train/test split of single dataset.
    Returns the best score and corresponding parameter list.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: best score (float) and corresponding list of parameters
    """

    # the best score result for a fit and its corresponding a parameter set
    best_score = sys.float_info.min
    best_params = None

    # create a parameter grid we'll use to parameterize the model at each iteration
    param_grid = ParameterGrid({'fit_intercept': [True, False], 'normalize': [True, False]})

    # iterate over each model parameterization
    for params in param_grid:
        model = LinearRegression(**params)
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        if score > best_score:
            best_score = score
            best_params = params

    return best_score, best_params


# ----------------------------------------------------------------------------------------------------------------------
def score_regression_ridge(x_train,
                           y_train,
                           x_test,
                           y_test):
    """
    Train and test using the ridge regression model using a train/test split of single dataset.
    Returns the best score and corresponding parameter list.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: best score (float) and corresponding list of parameters
    """

    # the best score result for a fit and its corresponding a parameter set
    best_score = sys.float_info.min
    best_params = None

    # create a parameter grid we'll use to parameterize the model at each iteration
    alphas = [0.25, 0.5, 1.0, 2.5, 5, 10]
    max_iterations = [None, 1, 2, 5, 10, 50, 100, 1000, 100000]
    tolerances = [0.00001, 0.001, 0.01, 0.1, 1, 2, 5, 10]
    solvers = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    param_grid = ParameterGrid({'alpha': alphas,
                                'max_iter': max_iterations,
                                'tol': tolerances,
                                'solver': solvers})

    # iterate over each model parameterization
    for params in param_grid:
        try:
            model = Ridge(**params)
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            if score > best_score:
                best_score = score
                best_params = params

        except Exception:
            _logger.exception('Failed to complete', exc_info=True)
            raise

    return best_score, best_params


# ----------------------------------------------------------------------------------------------------------------------
def score_regression_forest(x_train,
                            y_train,
                            x_test,
                            y_test):
    """
    Train and test using the random forest regression model using a train/test split of single dataset.
    Returns the best score and corresponding parameter list.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: best score (float) and corresponding list of parameters
    """

    # the best score result for a fit and its corresponding a parameter set
    best_score = sys.float_info.min
    best_params = None

    # create a parameter grid we'll use to parameterize the model at each iteration
    estimators = [10, 25, 100, 200]
    criteria = ['mse', 'mae']
    max_features = [1, 2, 3, 0.5, 0.75, 'auto', 'sqrt', 'log2']
    max_depths = [None, 1, 2, 3, 5, 10, 20]
    min_samples_splits = [2, 5, 10, 0.1, 0.25, 0.5, 0.75]
    min_samples_leafs = [1, 2, 5, 10, 0.1, 0.25, 0.5]
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
            _logger.info("\t\tFitting/scoring Random Forest with parameters: {p}".format(p=params))
            model = RandomForestRegressor(**params)
            model.fit(x_train, y_train.values.ravel())
            score = model.score(x_test, y_test)
            _logger.info("\t\t\tScore: {s}".format(s=score))
            if score > best_score:
                best_score = score
                best_params = params

        except Exception:
            _logger.exception('Failed to complete', exc_info=True)
            raise

    return best_score, best_params


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
                             level,
                             hemisphere=None):
    """
    Create a pandas DataFrame from variables of an xarray DataSet.

    :param dataset: xarray.DataSet
    :param variables: list of variables to be extracted from the DataSet and included in the resulting DataFrame
    :param level: the level index (all times, lats, and lons included at this indexed level)
    :param hemisphere: 'north', 'south', or None
    :return:
    """

    # the dataframe we'll populate and return
    df = pd.DataFrame()

    # slice the dataset down to a hemisphere, if specified
    if hemisphere is not None:
        if hemisphere == 'north':
            dataset = dataset.sel(lat=(dataset.lat >= 0))
        elif hemisphere == 'south':
            dataset = dataset.sel(lat=(dataset.lat < 0))
        else:
            raise ValueError("Unsupported hemisphere argument: {hemi}".format(hemi=hemisphere))

    # loop over each variable, adding each into the dataframe
    for var in variables:

        # if we have (time, lev, lat, lon), then use level parameter
        dimensions = dataset.variables[var].dims
        if dimensions == ('time', 'lev', 'lat', 'lon'):
            values = dataset[var].values[:, level, :, :]
        elif dimensions == ('time', 'lat', 'lon'):
            values = dataset[var].values[:, :, :]
        else:
            raise ValueError("Unsupported variable dimensions: {dims}".format(dims=dimensions))

        series = pd.Series(values.flatten())

        # add the series into the dataframe
        df[var] = series

    # make sure we have a generic index name
    df.index.rename('index', inplace=True)

    return df


# ----------------------------------------------------------------------------------------------------------------------
def split_hemispheres(features_dataset,
                      labels_dataset,
                      feature_vars,
                      label_vars,
                      level_ix):
    """
    Split the features and labels datasets into train and test arrays, using the northern hemisphere
    for training and the southern hemisphere for testing. Assumes a regular global grid with full
    northern and southern hemispheres.

    :param features_dataset: xarray.DataSet
    :param labels_dataset: xarray.DataSet
    :param feature_vars: list of variables to include from the features DataSet
    :param label_vars: list of variables to include from the labels DataSet
    :param level_ix: level coordinate index, assumes a 'lev' coordinate for all specified feature and label variables
    :return:
    """

    # make DataFrame from features, using the northern hemisphere for training data
    train_x = pull_vars_into_dataframe(features_dataset,
                                       feature_vars,
                                       level_ix,
                                       hemisphere='north')

    # make DataFrame from features, using the southern hemisphere for testing data
    test_x = pull_vars_into_dataframe(features_dataset,
                                      feature_vars,
                                      level_ix,
                                      hemisphere='south')

    # make DataFrame from labels, using the northern hemisphere for training data
    train_y = pull_vars_into_dataframe(labels_dataset,
                                       label_vars,
                                       level_ix,
                                       hemisphere='north')

    # make DataFrame from labels, using the southern hemisphere for testing data
    test_y = pull_vars_into_dataframe(labels_dataset,
                                      label_vars,
                                      level_ix,
                                      hemisphere='south')

    return train_x, test_x, train_y, test_y


# ----------------------------------------------------------------------------------------------------------------------
def score_models(dataset_features,
                 dataset_labels,
                 feature_vars,
                 label_vars,
                 split_on_hemispheres=False):
    """

    :param dataset_features:
    :param dataset_labels:
    :param feature_vars:
    :param label_vars:
    :param split_on_hemispheres: if True then use the northern hemisphere as the training set and
                the southern hemisphere as the testing dataset (50/50 split)
    :return:
    """

    # we assume each variable has multiple levels, and we loop over each
    for lev in range(dataset_features.lev.size):

        # get all features into a dataframe for this level, if not splitting on hemispheres
        if not split_on_hemispheres:

            df_features = pull_vars_into_dataframe(dataset_features,
                                                   feature_vars,
                                                   level=lev)

        # train/test for each label at this level
        for label in label_vars:

            # split into train/test datasets
            if split_on_hemispheres:

                train_x, test_x, train_y, test_y = split_hemispheres(dataset_features,
                                                                     dataset_labels,
                                                                     feature_vars,
                                                                     [label],
                                                                     level_ix=lev)

            else:

                # get the label data for this level into a dataframe
                df_labels = pull_vars_into_dataframe(dataset_labels,
                                                     [label],
                                                     level=lev)

                train_x, test_x, train_y, test_y = train_test_split(df_features,
                                                                    df_labels,
                                                                    test_size=0.25,
                                                                    random_state=4)

            # for this group of features/label perform some training/tests using various regression models
            _logger.info("Model results for features: {fs} and label: {lbl} at level {lvl}".format(fs=feature_vars,
                                                                                                   lbl=label,
                                                                                                   lvl=lev))

            # score the linear regression model using various parameters, log the best results
            best_score, best_params = score_regression_linear(train_x, train_y, test_x, test_y)
            _logger.info("LinearRegressor model results:")
            _logger.info("    Best parameter set: {params}".format(params=best_params))
            _logger.info("    Best score: {score}".format(score=best_score))

            # # only try K-Neighbors if we didn't get a decent score from linear regression
            # if best_score < 0.97:
            #     # score the K nearest neighbors regression model using various parameters, log the best results
            #     best_score, best_params = score_regression_kneighbors(train_x, train_y, test_x, test_y)
            #     _logger.info("KNeighborsRegressor model results:")
            #     _logger.info("    Best parameter set: {params}".format(params=best_params))
            #     _logger.info("    Best score: {score}".format(score=best_score))

            # # linear regression seems sufficient up to level 10 or so
            # if best_score < 0.97:
            #     # score the ridge regression model using various parameters, log the best results
            #     best_score, best_params = score_regression_ridge(train_x, train_y, test_x, test_y)
            #     _logger.info("Ridge model results:")
            #     _logger.info("    Best parameter set: {params}".format(params=best_params))
            #     _logger.info("    Best score: {score}".format(score=best_score))

            # only try Random Forest if we didn't get a decent score from linear regression
            if best_score < 0.97:
                # score the random forest regression model using various parameters, log the best results
                best_score, best_params = score_regression_forest(train_x, train_y, test_x, test_y)
                _logger.info("Random Forest model results:")
                _logger.info("    Best parameter set: {params}".format(params=best_params))
                _logger.info("    Best score: {score}".format(score=best_score))


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

        # confirm that we have datasets that match on the time, lev, lat, and lon dimension/coordinate
        if (ds_features.variables['time'].values != ds_labels.variables['time'].values).any():
            raise ValueError('Non-matching time values between feature and label datasets')
        if (ds_features.variables['lev'].values != ds_labels.variables['lev'].values).any():
            raise ValueError('Non-matching level values between feature and label datasets')
        if (ds_features.variables['lat'].values != ds_labels.variables['lat'].values).any():
            raise ValueError('Non-matching lat values between feature and label datasets')
        if (ds_features.variables['lon'].values != ds_labels.variables['lon'].values).any():
            raise ValueError('Non-matching lon values between feature and label datasets')

        # # TODO get initial year, month, and day from the datasets
        # init_year = 2000
        # init_month = 12
        # init_day = 27
        # timestamps = extract_timestamps(ds_features, init_year, init_month, init_day)

        # train/fit/score models using the dry features and corresponding labels
        features = ['PS', 'T', 'U', 'V']
        labels = ['PTTEND']
        score_models(ds_features,
                     ds_labels,
                     features,
                     labels,
                     split_on_hemispheres=True)

    except Exception:

        _logger.exception('Failed to complete', exc_info=True)
        raise
