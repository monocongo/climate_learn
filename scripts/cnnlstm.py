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
    for index, var in enumerate(variables):

        # if we have (time, lev, lat, lon), then use level parameter
        dimensions = dataset.variables[var].dims
        if dimensions == ('time', 'lev', 'lat', 'lon'):
            values = dataset[var].values[:, level, :, :]
        elif dimensions == ('time', 'lat', 'lon'):
            values = dataset[var].values[:, :, :]
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
    :return: numpy arrays with 4-D shape (times, lats, lons, vars)
    """

    # make array from features, using the northern hemisphere for training data
    train_x = pull_vars_into_array(features_dataset,
                                   feature_vars,
                                   level_ix,
                                   hemisphere='north')

    # make array from features, using the southern hemisphere for testing data
    test_x = pull_vars_into_array(features_dataset,
                                  feature_vars,
                                  level_ix,
                                  hemisphere='south')

    # make array from labels, using the northern hemisphere for training data
    train_y = pull_vars_into_array(labels_dataset,
                                   label_vars,
                                   level_ix,
                                   hemisphere='north')

    # make array from labels, using the southern hemisphere for testing data
    test_y = pull_vars_into_array(labels_dataset,
                                  label_vars,
                                  level_ix,
                                  hemisphere='south')

    return train_x, test_x, train_y, test_y


# ----------------------------------------------------------------------------------------------------------------------
def define_model_cnn_lstm(times, lats, lons, features, labels):
    """
    Create and return a model with CN and LSTM layers. Input data is expected to have
    shape (times, lats, lons, features) and output data will have shape (times, lats, lons, labels).

    *NOTE* Current model definition is specific to inputs with 128 lats and lons.

    :param times: time dimension of input/output 4-D array
    :param lats: latitude dimension of input/output 4-D array
    :param lons: longitude dimension of input/output 4-D array
    :param features: feature dimension of input 4-D array
    :param labels: label dimension of output 4-D array
    :return: CNN-LSTM model appropriate to the expected input/output arrays
    """

    # the model we'll build and return
    model = Sequential()

    # define the convolutional layers, wrapping each in a TimeDistributed layer
    model.add(TimeDistributed(Conv2D(filters=16, kernel_sze=(3, 3), activation='relu', padding='same'),
                              input_shape=(times, lats, lons, features)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(filters=32, kernel_sze=(3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(filters=32, kernel_sze=(3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))

    # add the LSTM layer
    model.add(LSTM(units=64, return_sequences=True))

    # reshape the result of the LSTM layer(s) to make it 2D and then use the combination
    # of UpSampling2D and Conv2D layers to get the original map's shape back
    model.add(TimeDistributed(Reshape((8, 8, 1))))
    model.add(TimeDistributed(UpSampling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(UpSampling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(UpSampling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(UpSampling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(labels, (3, 3), padding='same')))

    model.compile(optimizer='adam', loss='mse')

    return model


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
        if (ds_learn_features.variables['time'].values != ds_learn_labels.variables['time'].values).any():
            raise ValueError('Non-matching time values between feature and label datasets')
        if (ds_learn_features.variables['lev'].values != ds_learn_labels.variables['lev'].values).any():
            raise ValueError('Non-matching level values between feature and label datasets')
        if (ds_learn_features.variables['lat'].values != ds_learn_labels.variables['lat'].values).any():
            raise ValueError('Non-matching lat values between feature and label datasets')
        if (ds_learn_features.variables['lon'].values != ds_learn_labels.variables['lon'].values).any():
            raise ValueError('Non-matching lon values between feature and label datasets')

        # we assume the same number of levels in the prediction data as we have in the learning data
        if (ds_learn_features.variables['lev'].values != ds_predict_features.variables['lev'].values).any():
            raise ValueError('Non-matching level values between train and predict feature datasets')

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

        # reduce data into 128 lats and 128 lons


        # get the shape of the 4-D array we'll use for the predicted variable(s) data array
        out_size_time = ds_predict_features.time.size
        out_size_lev = ds_predict_features.lev.size
        out_size_lat = ds_predict_features.lat.size
        out_size_lon = ds_predict_features.lon.size
        prediction = np.empty(dtype=float, shape=(out_size_time, out_size_lev, out_size_lat, out_size_lon))

        # define the model
        model = define_model_cnn_lstm(out_size_time,
                                      int(out_size_lat / 2),
                                      out_size_lon,
                                      len(features),
                                      len(labels))

        # display the model summary
        model.summary()

        # loop over each level, keeping a record of the error rate for each level, for later visualization
        level_error_rates = {}
        for lev in range(out_size_lev):

            # split the data into train/test datasets using a north/south 50/50 split
            train_x, test_x, train_y, test_y = split_into_hemisphere_arrays(ds_learn_features,
                                                                            ds_learn_labels,
                                                                            features,
                                                                            labels,
                                                                            level_ix=lev)

            # scale the data into a (0..1) range since this will optimize the neural network's performance

            # initialize a 2-D list of lists (lats x lons) to score scalers for each lat/lon location
            scalers_x = [[None for _ in range(out_size_lon)] for _ in range(int(out_size_lat / 2))]
            scalers_y = [[None for _ in range(out_size_lon)] for _ in range(int(out_size_lat / 2))]

            # allocate arrays to hold scaled values
            train_x_scaled = np.copy(train_x)
            train_y_scaled = np.copy(train_y)
            test_x_scaled = np.copy(test_x)
            test_y_scaled = np.copy(test_y)

            # data is in 4-D with shape (times, lats, lons, vars), scalers can only work on 2-D arrays,
            # so for each lat/lon we scale the corresponding 2-D array, storing the scalers for later use
            # when scaling the data back to the original scale
            for lat in range(int(out_size_lat / 2)):  # divide by 2 since we're using hemispheres
                for lon in range(out_size_lon):
                    scaler_x = MinMaxScaler(feature_range=(0, 1))
                    scaler_y = MinMaxScaler(feature_range=(0, 1))
                    scalers_x[lat][lon] = scaler_x
                    scalers_y[lat][lon] = scaler_y
                    train_x_scaled[:, lat, lon, :] = scaler_x.fit_transform(train_x[:, lat, lon, :])
                    train_y_scaled[:, lat, lon, :] = scaler_y.fit_transform(train_y[:, lat, lon, :])
                    test_x_scaled[:, lat, lon, :] = scaler_x.transform(test_x[:, lat, lon, :])
                    test_y_scaled[:, lat, lon, :] = scaler_y.transform(test_y[:, lat, lon, :])

            # reshape to 5-D since this is required as input shape (initial dimension is batch?)
            shape = train_x_scaled.shape
            train_x_scaled = np.reshape(train_x_scaled, newshape=(1, shape[0], shape[1], shape[2], shape[3]))
            shape = train_y_scaled.shape
            train_y_scaled = np.reshape(train_y_scaled, newshape=(1, shape[0], shape[1], shape[2], shape[3]))

            # train the model for this level
            model.fit(train_x_scaled, train_y_scaled, epochs=2, shuffle=True, verbose=2)

            # get the new features from which we'll predict new label(s), using the same scaler as was used for training
            predict_x = pull_vars_into_array(ds_predict_features,
                                             features,
                                             lev)

            # allocate array to hold scaled values
            predict_x_scaled = np.copy(predict_x)

            # data is in 4-D with shape (times, lats, lons, vars), scalers can only work on 2-D arrays,
            # so for each lat/lon we scale the corresponding 2-D array, storing the scalers for later use
            # when scaling the data back to the original scale
            for lat in range(out_size_lat):
                for lon in range(out_size_lon):
                    scaler_x = scalers_x[lat][lon]
                    predict_x_scaled[:, lat, lon, :] = scaler_x.fit_transform(predict_x[:, lat, lon, :])

            # use the model to predict label values from new inputs
            predict_y_scaled = model.predict(predict_x_scaled, verbose=1)

            # unscale the data, using the same scaler as was used for training

            # allocate array to hold scaled values
            predict_y = np.copy(predict_y_scaled)

            # data is in 4-D with shape (times, lats, lons, vars), scalers can only work on 2-D arrays,
            # so for each lat/lon we scale the corresponding 2-D array, storing the scalers for later use
            # when scaling the data back to the original scale
            for lat in range(out_size_lat):
                for lon in range(out_size_lon):
                    scaler_y = scalers_y[lat][lon]
                    predict_y[:, lat, lon, :] = scaler_y.inverse_transform(predict_y_scaled[:, lat, lon, :])

            # write the predicted values for the level into the predicted label's data array
            prediction[:, lev, :, :] = np.reshape(predict_y, newshape=(out_size_time, out_size_lat, out_size_lon))

        # copy the prediction features dataset since the predicted label(s) should share the same coordinates, etc.
        ds_predict_labels = ds_predict_features.copy(deep=True)

        # remove all non-label data variables from the predictions dataset
        for var in ds_predict_labels.data_vars:
            if var not in labels:
                ds_predict_labels.drop(var)

        # create a new variable to contain the predicted label, assign it into the prediction dataset
        predicted_label_var = xr.Variable(dims=('time', 'lev', 'lat', 'lon'),
                                          data=prediction,
                                          attrs=ds_learn_labels[labels[0]].attrs)
        ds_predict_labels[labels[0]] = predicted_label_var

        # write the predicted label(s)' dataset as NetCDF
        ds_predict_labels.to_netcdf(args.predict_labels)

        # placeholder for debugging steps
        pass

    except Exception:

        _logger.exception('Failed to complete', exc_info=True)
        raise
