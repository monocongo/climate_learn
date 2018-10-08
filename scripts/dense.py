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
def define_model_dense(num_features,
                       num_labels):
    """
    Define a model using densely-connected neural network layers.

    :param num_features: the number of features (input attributes)
    :param num_labels: the number of labels the model should predict (output values)
    :return: a Keras neural network model that uses densely-connected layers
    """

    # define the model
    dense_model = Sequential()

    # add a single fully-connected hidden layer with the same number of neurons as input attributes (features)
    dense_model.add(Dense(num_features, input_dim=num_features, activation='relu'))

    # output layer uses no activation function since we are interested
    # in predicting numerical values directly without transform
    dense_model.add(Dense(num_labels))

    # compile the model using the ADAM optimization algorithm and a mean squared error loss function
    dense_model.compile(optimizer='adam', loss='mse')

    return dense_model


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

        # get the shape of the 4-D array we'll use for the predicted variable(s) data array
        out_size_time = ds_predict_features.time.size
        out_size_lev = ds_predict_features.lev.size
        out_size_lat = ds_predict_features.lat.size
        out_size_lon = ds_predict_features.lon.size
        prediction = np.empty(dtype=float, shape=(out_size_time, out_size_lev, out_size_lat, out_size_lon))

        # define the model
        model = define_model_dense(len(features), len(labels))

        # display the model summary
        model.summary()

        # loop over each level, keeping a record of the error rate for each level, for later visualization
        level_error_rates = {}
        for lev in range(out_size_lev):

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

            # train the model for this level
            model.fit(train_x_scaled, train_y_scaled, epochs=2, shuffle=True, verbose=2)

            # evaluate the model's fit
            level_error_rates[lev] = model.evaluate(test_x_scaled, test_y_scaled)

            # get the new features from which we'll predict new label(s), using the same scaler as was used for training
            predict_x = pull_vars_into_dataframe(ds_predict_features,
                                                 features,
                                                 lev)
            predict_x_scaled = scaler_x.transform(predict_x)

            # use the model to predict label values from new inputs
            predict_y_scaled = model.predict(predict_x_scaled, verbose=1)

            # unscale the data, using the same scaler as was used for training
            predict_y = scaler_y.inverse_transform(predict_y_scaled)

            # write the predicted values for the level into the predicted label's data array
            prediction[:, lev, :, :] = np.reshape(predict_y, newshape=(out_size_time, out_size_lat, out_size_lon))

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

        # placeholder for debugging steps
        pass

    except Exception:

        _logger.exception('Failed to complete', exc_info=True)
        raise
