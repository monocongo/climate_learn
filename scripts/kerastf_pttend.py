import argparse
import logging

from keras.models import Sequential
from keras.layers import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xarray as xr

# ----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)


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
if __name__ == '__main__':
    """
    This module is used to showcase ML modeling of the climate using scikit-learn, using NCAR CAM files as input.
    """

    try:

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--learn_features",
                            help="NetCDF files containing flow variables used to train/test the model",
                            nargs='*',
                            required=True)
        parser.add_argument("--learn_labels",
                            help="NetCDF files containing time tendency forcing variables used to train/test the model",
                            nargs='*',
                            required=True)
        parser.add_argument("--predict_features",
                            help="NetCDF files containing flow variables used as prediction inputs",
                            nargs='*',
                            required=True)
        parser.add_argument("--predict_labels",
                            help="NetCDF file to contain predicted time tendency forcing variables",
                            nargs='*',
                            required=True)
        args = parser.parse_args()

        # train/fit/score models using the dry features and corresponding labels
        features = ['PS', 'T', 'U', 'V']
        labels = ['PTTEND']

        # open the features (flows) and labels (tendencies) as xarray DataSets
        ds_learn_features = xr.open_mfdataset(paths=args.learn_features,
                                              data_vars=features)
        ds_learn_labels = xr.open_mfdataset(paths=args.learn_labels,
                                            data_vars=labels)
        ds_predict_features = xr.open_mfdataset(paths=args.predict_features,
                                                data_vars=features)

        # confirm that we have datasets that match on the time, lev, lat, and lon dimension/coordinate
        if (ds_learn_features.variables['time'].values != ds_learn_labels.variables['time'].values).any():
            raise ValueError('Non-matching time values between feature and label datasets')
        if (ds_learn_features.variables['lev'].values != ds_learn_labels.variables['lev'].values).any():
            raise ValueError('Non-matching level values between feature and label datasets')
        if (ds_learn_features.variables['lat'].values != ds_learn_labels.variables['lat'].values).any():
            raise ValueError('Non-matching lat values between feature and label datasets')
        if (ds_learn_features.variables['lon'].values != ds_learn_labels.variables['lon'].values).any():
            raise ValueError('Non-matching lon values between feature and label datasets')

        # get the shape of the 4-D array we'll use for the predicted variable(s) data array
        size_time = ds_learn_features.time.size
        size_lev = ds_learn_features.lev.size
        size_lat = ds_learn_features.lat.size
        size_lon = ds_learn_features.lon.size
        prediction = np.empty(dtype=float, shape=(size_time, size_lev, size_lat, size_lon))

        # define the model
        model = Sequential()
        model.add(Dense(50, input_dim=len(features), activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # loop over each level, keeping a record of the error rate for each level, for later visualization
        level_error_rates = {}
        for lev in range(size_lev):

            # split the data into train/test datasets using a north/south 50/50 split
            train_x, test_x, train_y, test_y = split_hemispheres(ds_learn_features,
                                                                 ds_learn_labels,
                                                                 features,
                                                                 labels,
                                                                 level_ix=lev)

            # scale the data into a small range since this will optimize the neural network's performance
            scaler_x = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            train_x_scaled = scaler_x.fit_transform(train_x)
            train_y_scaled = scaler_y.fit_transform(train_y)
            test_x_scaled = scaler_x.transform(test_x)
            test_y_scaled = scaler_y.transform(test_y)
            # train_x_scaled, train_y_scaled = scaler.fit_transform(train_x, train_y)
            # test_x_scaled, test_y_scaled = scaler.transform(test_x, test_y)

            # train the model for this level
            model.fit(train_x_scaled, train_y_scaled, epochs=4, shuffle=True, verbose=2)

            # evaluate the model's performance
            level_error_rates[lev] = model.evaluate(test_x_scaled, test_y_scaled, verbose=0)

            # get the new features from which we'll predict new label(s), using the same scaler as was used for training
            predict_x = pull_vars_into_dataframe(ds_predict_features,
                                                 features,
                                                 lev)
            predict_x_scaled = scaler_x.transform(predict_x)

            # use the model to predict label values from new inputs
            predict_y_scaled = model.predict(predict_x_scaled)[0][0]

            # unscale the data, using the same scaler as was used for training
            predict_y = scaler_y.inverse_transform(predict_y_scaled)

            # write the predicted values for the level into the predicted label's data array
            prediction[:, lev, :, :] = np.reshape(predict_y, shape=(size_time, 1, size_lat, size_lon))

        # copy the prediction features dataset since the predicted label(s) should share the same coordinates, etc.
        ds_predict_labels = ds_predict_features.copy(data={labels[0]: prediction})

        # remove all non-label variables from the dataset copy
        for var in ds_predict_labels.variables:
            if var not in labels:
                ds_predict_features.drop(var)

        # write the predicted label(s)' dataset as NetCDF
        ds_predict_labels.to_netcdf(args.predict_labels)

        # placeholder for debugging steps
        pass

    except Exception:

        _logger.exception('Failed to complete', exc_info=True)
        raise
