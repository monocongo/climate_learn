import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler

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

# function to perform scaling
def scale_4d(features_train,
             features_predict,
             labels_train,
             scalers_feature,
             scalers_label):
    
    # make new arrays to contain the scaled values we'll return
    scaled_features_train = np.empty(shape=features_train.shape)
    scaled_features_predict = np.empty(shape=features_predict.shape)
    scaled_labels_train = np.empty(shape=labels_train.shape)
    
    # data is 4-D with shape (times, lats, lons, vars), scalers can only work on 2-D arrays,
    # so for each feature we scale the corresponding 3-D array of values after flattening it,
    # then reshape back into the original shape
    for feature_ix in range(features_train.shape[-1]):
        scaler = scalers_feature[feature_ix]
        feature_train = features_train[:, :, :, feature_ix].flatten().reshape(-1, 1)
        feature_predict = features_predict[:, :, :, feature_ix].flatten().reshape(-1, 1)
        scaled_train = scaler.fit_transform(feature_train)
        scaled_predict = scaler.fit_transform(feature_predict)
        reshaped_scaled_train = np.reshape(scaled_train, newshape=(size_times_train, size_lat, size_lon))
        reshaped_scaled_predict = np.reshape(scaled_predict, newshape=(size_times_predict, size_lat, size_lon))
        scaled_features_train[:, :, :, feature_ix] = reshaped_scaled_train
        scaled_features_predict[:, :, :, feature_ix] = reshaped_scaled_predict
    for label_ix in range(len(labels)):
        scaler = scalers_label[label_ix]
        label_train = labels_train[:, :, :, label_ix].flatten().reshape(-1, 1)
        scaled_train = scaler.fit_transform(label_train)
        reshaped_scaled_train = np.reshape(scaled_train, newshape=(size_times_train, size_lat, size_lon))
        scaled_labels_train[:, :, :, label_ix] = reshaped_scaled_train
    
    # return the scaled values as well as the scalers that have been fitted to the data
    return scaled_features_train, scaled_features_predict, scaled_labels_train, scalers_feature, scalers_label

