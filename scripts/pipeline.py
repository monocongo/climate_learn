import argparse
from collections import OrderedDict

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import xarray as xr


# ------------------------------------------------------------------------------
class Scaler(TransformerMixin):

    def __init__(self, features):

        # initialize an ordered dict to store scalers for each feature
        self.scalers = OrderedDict().fromkeys(features, MinMaxScaler(feature_range=(0, 1)))

    def transform(self, values):
        """
        Transforms a 4-D array of values, expected to have shape:
        (times, lats, lons, vars).

        :param values:
        :return:
        """

        # make new arrays to contain the scaled values we'll return
        scaled_features = np.empty(shape=values.shape)

        # data is 4-D with shape (times, lats, lons, vars), scalers can only
        # work on 2-D arrays, so for each variable we scale the corresponding
        # 3-D array of values after flattening it, then reshape back into
        # the original shape
        var_ix = 0
        for variable, scaler in self.scalers.items():
            variable = values[:, :, :, var_ix].flatten().reshape(-1, 1)
            scaled_feature = scaler.fit_transform(variable)
            reshaped_scaled_feature = np.reshape(scaled_feature,
                                                 newshape=(values.shape[0],
                                                           values.shape[1],
                                                           values.shape[2]))
            scaled_features[:, :, :, var_ix] = reshaped_scaled_feature
            var_ix += 1

        # return the scaled values (the scalers have been fitted to the data)
        return scaled_features

    def fit(self, x=None):

        return self


# ------------------------------------------------------------------------------
def extract_data_array(dataset,
                       variables,
                       lev):
    # allocate the array
    arr = np.empty(shape=[dataset.time.size,
                          dataset.lat.size,
                          dataset.lon.size,
                          len(variables)],
                   dtype=np.float64)

    # for each variable we'll extract the values
    for var_index, var in enumerate(variables):

        # if we have (time, lev, lat, lon), then use level parameter
        dimensions = dataset.variables[var].dims
        if dimensions == ('time', 'lev', 'lat', 'lon'):
            values = dataset[var].values[:, lev, :, :]
        elif dimensions == ('time', 'lat', 'lon'):
            values = dataset[var].values[:, :, :]
        else:
            raise ValueError("Unsupported variable dimensions: {dims}".format(dims=dimensions))

        # add the values into the array at the variable's position
        arr[:, :, :, var_index] = values

    return arr


# ------------------------------------------------------------------------------
def extract_features_labels(netdcf_features,
                            netcdf_labels,
                            feature_vars,
                            label_vars,
                            level=0):
    """
    Extracts feature and label data from specified NetCDF files for a single level as numpy arrays.

    The feature and label NetCDFs are expected to have matching time, level, lat, and lon coordinate variables.

    Returns two arrays: the first for features and the second for labels. Arrays will have shape (time, lat, lon, var),
    where var is the number of feature or label variables. For example if the dimensions of feature data variables in
    the NetCDF is (time: 360, lev: 26, lat: 120, lon: 180) and the features specified are ["T", "U"] then the resulting
    features array will have shape (360, 120, 180, 2), with the first feature variable "T" corresponding to array[:, :, :, 0]
    and the second feature variable "U" corresponding to array[:, :, :, 1].

    :param netdcf_features: one or more NetCDF files containing feature variables, can be single file or list
    :param netdcf_features: one or more NetCDF files containing label variables, can be single file or list
    :param feature_vars: list of feature variable names to be extracted from the features NetCDF
    :param label_vars: list of label variable names to be extracted from the labels NetCDF
    :param level: index of the level to be extracted (all times/lats/lons at this level for each feature/label variable)
    :return: two 4-D numpy arrays, the first for features and the second for labels
    """

    # open the features (flows) and labels (tendencies) as xarray DataSets
    ds_features = xr.open_mfdataset(paths=netdcf_features)
    ds_labels = xr.open_mfdataset(paths=netcdf_labels)

    # confirm that we have datasets that match on the time, lev, lat, and lon dimension/coordinate
    if np.any(ds_features.variables['time'].values != ds_labels.variables['time'].values):
        raise ValueError('Non-matching time values between feature and label datasets')
    if np.any(ds_features.variables['lev'].values != ds_labels.variables['lev'].values):
        raise ValueError('Non-matching level values between feature and label datasets')
    if np.any(ds_features.variables['lat'].values != ds_labels.variables['lat'].values):
        raise ValueError('Non-matching lat values between feature and label datasets')
    if np.any(ds_features.variables['lon'].values != ds_labels.variables['lon'].values):
        raise ValueError('Non-matching lon values between feature and label datasets')

    # extract feature and label arrays at the specified level
    array_features = extract_data_array(ds_features, feature_vars, level)
    array_labels = extract_data_array(ds_labels, label_vars, level)

    return array_features, array_labels


# ------------------------------------------------------------------------------
def create_dense_model():

    # define the model
    dense_model = Sequential()

    # add a fully-connected hidden layer with the same number of neurons as input attributes (features)
    dense_model.add(Dense(len(features), input_dim=len(features), activation='relu'))

    # add a fully-connected hidden layer with the twice the number of neurons as input attributes (features)
    dense_model.add(Dense(len(features) * 2, activation='relu'))

    # output layer uses no activation function since we are interested
    # in predicting numerical values directly without transform
    dense_model.add(Dense(len(labels)))

    # compile the model using the ADAM optimization algorithm and a mean squared error loss function
    dense_model.compile(optimizer='adam', loss='mse')

    return dense_model


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_nc",
        help="NetCDF containing features",
        required=True,
    )
    parser.add_argument(
        "--feature_names",
        help="Name(s) of feature variables to extract (in order) from the features NetCDF",
        required=True,
    )
    parser.add_argument(
        "--labels_nc",
        help="NetCDF containing labels",
        required=True,
    )
    parser.add_argument(
        "--label_names",
        help="Name(s) of label variables to extract (in order) from the labels NetCDF",
        required=True,
    )
    args = parser.parse_args()

    features, labels = extract_features_labels(args.features_nc,
                                               args.labels_nc,
                                               args.feature_names,
                                               args.label_names,
                                               level=0)

    X_train, X_test, y_train, y_test = train_test_split(features, labels)

    pipeline = Pipeline(steps=[
        ('scaler', Scaler(args.feature_names)),
        ('model', create_dense_model()),
    ])

    pipeline.fit(X_train, y_train)
    print('Testing score: ', pipeline.score(X_test, y_test))
    y_predict = pipeline.predict(X_test)

    param_to_test_0 = np.arange(1, 11)
    param_to_test_1 = 2.0 ** np.arange(-6, +6)
    params = {'param_0': param_to_test_0,
              'param_1': param_to_test_1}
    grid_search = GridSearchCV(pipeline, params, verbose=1).fit(X_train, y_train)
    print('Final score is: ', grid_search.score(X_test, y_test))
