import argparse
import logging

from keras.models import Sequential
from keras.layers import Conv3D, Dense, MaxPooling3D
import numpy as np
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
    Create a pandas DataFrame from the specified variables of an xarray DataSet.

    :param dataset: xarray.DataSet
    :param variables: list of variables to be extracted from the DataSet and included in the resulting DataFrame
    :param level: the level index (all times, lats, and lons included from this indexed level)
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
def split_into_hemisphere_dfs(features_dataset,
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
    train_features = pull_vars_into_dataframe(features_dataset,
                                              feature_vars,
                                              level_ix,
                                              hemisphere='north')

    # make DataFrame from features, using the southern hemisphere for testing data
    test_features = pull_vars_into_dataframe(features_dataset,
                                             feature_vars,
                                             level_ix,
                                             hemisphere='south')

    # make DataFrame from labels, using the northern hemisphere for training data
    train_labels = pull_vars_into_dataframe(labels_dataset,
                                            label_vars,
                                            level_ix,
                                            hemisphere='north')

    # make DataFrame from labels, using the southern hemisphere for testing data
    test_labels = pull_vars_into_dataframe(labels_dataset,
                                           label_vars,
                                           level_ix,
                                           hemisphere='south')

    return train_features, test_features, train_labels, test_labels


# ----------------------------------------------------------------------------------------------------------------------
def define_model_cnn(num_times,
                     num_lats,
                     num_lons,
                     num_features,
                     num_labels):
    """
    Define a model using convolutional neural network layers.

    Input data is expected to have shape (1, times, lats, lons, features) and output data
    will have shape (1, times, lats, lons, labels).

    :param num_times: the number of times in the input's time dimension
    :param num_lats: the number of lats in the input's lat dimension
    :param num_lons: the number of lons in the input's lon dimension
    :param num_features: the number of features (input attributes) in the input's channel dimension
    :param num_labels: the number of labels (output attributes) in the output's channel dimension
    :return: a Keras neural network model that uses convolutional layers
    """

    # define the model
    cnn_model = Sequential()

    # add an initial 3-D convolutional layer
    model.add(Conv3D(filters=32,
                     kernel_size=(3, 3, 3),
                     activation="relu",
                     data_format="channels_last",
                     input_shape=(num_times, num_lats, num_lons, num_features),
                     padding='same'))

    # add a pooling layer
    cnn_model.add(MaxPooling3D(pool_size=(2, 2, 2),
                               strides=(1, 1, 1),
                               padding='valid',
                               data_format="channels_last"))

    # output layer uses no activation function since we are interested
    # in predicting numerical values directly without transform
    cnn_model.add(Dense(num_labels))

    # compile the model using the ADAM optimization algorithm and a mean squared error loss function
    cnn_model.compile(optimizer='adam', loss='mse')

    return cnn_model


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This module is used to showcase ML modeling of the climate using neural networks, 
    initially using NCAR CAM files as inputs.
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
        args = parser.parse_args()

        # train/fit/score models using the dry features and corresponding labels
        features = ['PS', 'T', 'U', 'V']
        labels = ['PTTEND']

        # open the features (flows) and labels (tendencies) as xarray DataSets
        ds_learn_features = xr.open_mfdataset(paths=args.learn_features)
        ds_learn_labels = xr.open_mfdataset(paths=args.learn_labels)

        # confirm that we have learning datasets that match on the time, lev, lat, and lon dimension/coordinate
        if np.any(ds_learn_features.variables['time'].values != ds_learn_labels.variables['time'].values):
            raise ValueError('Non-matching time values between feature and label datasets')
        if np.any(ds_learn_features.variables['lev'].values != ds_learn_labels.variables['lev'].values):
            raise ValueError('Non-matching level values between feature and label datasets')
        if np.any(ds_learn_features.variables['lat'].values != ds_learn_labels.variables['lat'].values):
            raise ValueError('Non-matching lat values between feature and label datasets')
        if np.any(ds_learn_features.variables['lon'].values != ds_learn_labels.variables['lon'].values):
            raise ValueError('Non-matching lon values between feature and label datasets')

        # trim out all non-relevant data variables from the datasets
        for var in ds_learn_features.data_vars:
            if var not in features:
                ds_learn_features = ds_learn_features.drop(var)
        for var in ds_learn_labels.data_vars:
            if var not in labels:
                ds_learn_labels = ds_learn_labels.drop(var)

        # get data dimension sizes
        size_time = ds_learn_features.time.size
        size_lev = ds_learn_features.lev.size
        size_lat = ds_learn_features.lat.size
        size_lon = ds_learn_features.lon.size

        # define the model
        model = define_model_cnn(size_time, size_lat, size_lon, len(features), len(labels))

        # display the model summary
        model.summary()

        # loop over each level, keeping a record of the error rate for each level, for later visualization
        level_error_rates = {}
        for lev in range(size_lev):

            # split the data into train/test datasets using a north/south 50/50 split
            train_x, test_x, train_y, test_y = split_into_hemisphere_dfs(ds_learn_features,
                                                                         ds_learn_labels,
                                                                         features,
                                                                         labels,
                                                                         level_ix=lev)

            # scale the data into a (0..1) range since this will optimize the neural network's performance
            scaler_x = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))

            # fit to both hemispheres
            scaler_x.fit(pd.concat([train_x, test_x]))
            scaler_y.fit(pd.concat([train_y, test_y]))

            # perform scaling
            train_x_scaled = scaler_x.transform(train_x)
            train_y_scaled = scaler_y.transform(train_y)
            test_x_scaled = scaler_x.transform(test_x)
            test_y_scaled = scaler_y.transform(test_y)

            # reshape the data arrays into the model's expected input shape
            train_x = np.reshape(train_x_scaled, newshape=(size_time, size_lat, size_lon, len(features)))
            train_y = np.reshape(train_y_scaled, newshape=(size_time, size_lat, size_lon, len(labels)))
            test_x = np.reshape(test_x_scaled, newshape=(size_time, size_lat, size_lon, len(features)))
            test_y = np.reshape(test_y_scaled, newshape=(size_time, size_lat, size_lon, len(labels)))

            # train the model for this level
            model.fit(train_x, train_y, epochs=4, shuffle=True, verbose=2)

            # evaluate the model's fit
            level_error_rates[lev] = model.evaluate(test_x_scaled, test_y_scaled)

        # placeholder for debugging steps
        pass

        # visualization code (matplotlib/seaborn) here for analysis of the evaluations performed above

    except Exception:

        _logger.exception('Failed to complete', exc_info=True)
        raise
