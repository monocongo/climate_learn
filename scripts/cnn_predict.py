import argparse
import logging

from keras.models import Sequential
from keras.layers import Conv3D, Dense
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import xarray as xr

# ----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
def pull_vars_into_array(dataset,
                         variables,
                         level,
                         hemisphere=None):
    """
    Create a Numpy array from the specified variables of an xarray DataSet.

    :param dataset: xarray.DataSet
    :param variables: list of variables to be extracted from the DataSet and included in the resulting DataFrame
    :param level: the level index (all times, lats, and lons included from this indexed level)
    :param hemisphere: 'north', 'south', or None
    :return: an array with shape (ds.time.size, ds.lat.size, ds.lon.size, len(variables)) and dtype float
    """

    # slice the dataset down to a hemisphere, if specified
    if hemisphere is not None:
        if hemisphere == 'north':
            dataset = dataset.sel(lat=(dataset.lat >= 0))
        elif hemisphere == 'south':
            dataset = dataset.sel(lat=(dataset.lat < 0))
        else:
            raise ValueError("Unsupported hemisphere argument: {hemi}".format(hemi=hemisphere))

    # the array we'll populate and return
    arr = np.empty(shape=[dataset.time.size, dataset.lat.size, dataset.lon.size, len(variables)], dtype=float)

    # loop over each variable, adding each into the dataframe
    for index, var_name in enumerate(variables):

        # if we have (time, lev, lat, lon), then use level parameter
        dimensions = dataset.variables[var_name].dims
        if dimensions == ('time', 'lev', 'lat', 'lon'):
            values = dataset[var_name].values[:, level, :, :]
        elif dimensions == ('time', 'lat', 'lon'):
            values = dataset[var_name].values[:, :, :]
        else:
            raise ValueError("Unsupported variable dimensions: {dims}".format(dims=dimensions))

        # add the values into the array at the variable's position
        arr[:, :, :, index] = values

    return arr


# ----------------------------------------------------------------------------------------------------------------------
def split_into_hemisphere_arrays(features_dataset,
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
    train_features = pull_vars_into_array(features_dataset,
                                          feature_vars,
                                          level_ix,
                                          hemisphere='north')

    # make DataFrame from features, using the southern hemisphere for testing data
    test_features = pull_vars_into_array(features_dataset,
                                         feature_vars,
                                         level_ix,
                                         hemisphere='south')

    # make DataFrame from labels, using the northern hemisphere for training data
    train_labels = pull_vars_into_array(labels_dataset,
                                        label_vars,
                                        level_ix,
                                        hemisphere='north')

    # make DataFrame from labels, using the southern hemisphere for testing data
    test_labels = pull_vars_into_array(labels_dataset,
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
    cnn_model.add(Conv3D(filters=32,
                         kernel_size=(3, 3, 3),
                         activation="relu",
                         data_format="channels_last",
                         input_shape=(num_times, num_lats, num_lons, num_features),
                         padding='same'))

    # add a fully-connected hidden layer with twice the number of neurons as input attributes (features)
    cnn_model.add(Dense(num_features * 2, activation='relu'))

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
        parser.add_argument("--predict_features",
                            help="NetCDF files containing flow variables used as prediction inputs",
                            nargs='*',
                            required=True)
        parser.add_argument("--predict_labels",
                            help="NetCDF file to contain predicted time tendency forcing variables",
                            required=True)
        args = parser.parse_args()

        # train/fit/score models using the dry features and corresponding labels
        features = ['PS', 'T', 'U', 'V']
        labels = ['PTTEND']

        # open the features (flows) and labels (tendencies) as xarray DataSets
        ds_learn_features = xr.open_mfdataset(paths=args.learn_features)
        ds_learn_labels = xr.open_mfdataset(paths=args.learn_labels)
        ds_predict_features = xr.open_mfdataset(paths=args.predict_features)

        # confirm that we have learning datasets that match on the time, lev, lat, and lon dimension/coordinate
        if np.any(ds_learn_features.variables['time'].values != ds_learn_labels.variables['time'].values):
            raise ValueError('Non-matching time values between feature and label datasets')
        if np.any(ds_learn_features.variables['lev'].values != ds_learn_labels.variables['lev'].values):
            raise ValueError('Non-matching level values between feature and label datasets')
        if np.any(ds_learn_features.variables['lat'].values != ds_learn_labels.variables['lat'].values):
            raise ValueError('Non-matching lat values between feature and label datasets')
        if np.any(ds_learn_features.variables['lon'].values != ds_learn_labels.variables['lon'].values):
            raise ValueError('Non-matching lon values between feature and label datasets')

        # confirm that the learning and prediction datasets match on the lev, lat, and lon dimension/coordinate
        # it's likely that we'll use more times for training than for prediction, so we ignore those differences
        if np.any(ds_learn_features.variables['lev'].values != ds_predict_features.variables['lev'].values):
            raise ValueError('Non-matching level values between train and predict feature datasets')
        if np.any(ds_learn_features.variables['lat'].values != ds_predict_features.variables['lat'].values):
            raise ValueError('Non-matching lat values between train and predict feature datasets')
        if np.any(ds_learn_features.variables['lon'].values != ds_predict_features.variables['lon'].values):
            raise ValueError('Non-matching lon values between train and predict feature datasets')

        # trim out all non-relevant data variables from the datasets
        for var in ds_learn_features.data_vars:
            if var not in features:
                ds_learn_features = ds_learn_features.drop(var)
        for var in ds_learn_labels.data_vars:
            if var not in labels:
                ds_learn_labels = ds_learn_labels.drop(var)
        for var in ds_predict_features.data_vars:
            if var not in features:
                ds_predict_features = ds_predict_features.drop(var)

        # get data dimension sizes
        size_time = ds_learn_features.time.size
        size_lev = ds_learn_features.lev.size
        size_lat = ds_learn_features.lat.size
        size_lon = ds_learn_features.lon.size

        # allocate an array for a single predicted variable based on the above dimension sizes
        # TODO do this for all labels, for the case where there are multiple labels to predict
        prediction = np.empty(dtype=float, shape=(ds_predict_features.time.size, size_lev, size_lat, size_lon))

        # define the model
        model = define_model_cnn(size_time, size_lat, size_lon, len(features), len(labels))

        # display the model summary
        model.summary()

        # initialize a list to store scalers for each feature/label
        scalers_x = [MinMaxScaler(feature_range=(0, 1))] * len(features)
        scalers_y = [MinMaxScaler(feature_range=(0, 1))] * len(labels)

        # loop over each level, keeping a record of the error rate for each level, for later visualization
        for lev in range(size_lev):

            # get the array of features for this level (all times/lats/lons)
            train_x = pull_vars_into_array(ds_learn_features,
                                           features,
                                           lev)

            # get the array of labels for this level (all times/lats/lons)
            train_y = pull_vars_into_array(ds_learn_labels,
                                           labels,
                                           lev)

            # get the new features from which we'll predict new label(s)
            predict_x = pull_vars_into_array(ds_predict_features,
                                             features,
                                             lev)

            # data is 4-D with shape (times, lats, lons, vars), scalers can only work on 2-D arrays,
            # so for each feature we scale the corresponding 3-D array of values after flattening it,
            # then reshape back into the original shape
            for feature_ix in range(len(features)):
                scaler = scalers_x[feature_ix]
                feature_train = train_x[:, :, :, feature_ix].flatten().reshape(-1, 1)
                feature_predict = predict_x[:, :, :, feature_ix].flatten().reshape(-1, 1)
                scaled_train = scaler.fit_transform(feature_train)
                scaled_predict = scaler.fit_transform(feature_predict)
                reshaped_scaled_train = np.reshape(scaled_train, newshape=(size_time, size_lat, size_lon))
                reshaped_scaled_predict = np.reshape(scaled_predict, newshape=(predict_x.shape[0], size_lat, size_lon))
                train_x[:, :, :, feature_ix] = reshaped_scaled_train
                predict_x[:, :, :, feature_ix] = reshaped_scaled_predict
            for label_ix in range(len(labels)):
                scaler = scalers_y[label_ix]
                label_train = train_y[:, :, :, label_ix].flatten().reshape(-1, 1)
                scaled_train = scaler.fit_transform(label_train)
                reshaped_scaled_train = np.reshape(scaled_train, newshape=(size_time, size_lat, size_lon))
                train_y[:, :, :, label_ix] = reshaped_scaled_train

            # reshape from 4-D to 5-D (expected model input shape)
            train_x = np.reshape(train_x, newshape=(1, ) + train_x.shape)
            train_y = np.reshape(train_y, newshape=(1, ) + train_y.shape)
            predict_x = np.reshape(predict_x, newshape=(1, ) + predict_x.shape)

            # train the model for this level
            model.fit(train_x, train_y, shuffle=True, epochs=8, verbose=2)

            # use the model to predict label values from new inputs
            predict_y = model.predict(predict_x, verbose=1)

            # unscale the predicted labels data, using the same scalers as were used for training

            for label_ix in range(len(labels)):
                scaler = scalers_y[label_ix]
                label_predict = predict_y[:, :, :, label_ix].flatten().reshape(-1, 1)
                unscaled_predict = scaler.inverse_transform(label_predict)
                reshaped_unscaled_predict = np.reshape(unscaled_predict, newshape=(size_time, size_lat, size_lon))
                predict_y[:, :, :, label_ix] = reshaped_unscaled_predict

            # write the predicted values for the level into the predicted label's data array
            prediction[:, lev, :, :] = np.reshape(predict_y, newshape=(size_time, size_lat, size_lon))

        # copy the prediction features dataset since the predicted label(s) should share the same coordinates, etc.
        ds_predict_labels = ds_predict_features.copy(deep=True)

        # remove all non-label data variables from the predictions dataset
        for var in ds_predict_labels.data_vars:
            if var not in labels:
                ds_predict_labels = ds_predict_labels.drop(var)

        # create a new variable to contain the predicted label, assign it into the prediction dataset
        predicted_label_var = xr.Variable(dims=('time', 'lev', 'lat', 'lon'),
                                          data=prediction,
                                          attrs=ds_learn_labels[labels[0]].attrs)
        ds_predict_labels[labels[0]] = predicted_label_var

        # write the predicted label(s)' dataset as NetCDF
        ds_predict_labels.to_netcdf(args.predict_labels)

        # place holder for debugging step
        pass

    except Exception:

        _logger.exception('Failed to complete', exc_info=True)
        raise
