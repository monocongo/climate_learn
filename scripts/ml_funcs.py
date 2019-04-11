import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler

def extract_data_array(dataset,
                       variables,
                       lev,
                       nf):

    # allocate the array
    arr = np.empty(shape=[dataset.time.size, 
                          dataset.lat.size, 
                          dataset.lon.size, 
                          len(variables)],
                   dtype=np.float64)
    
    # for each variable we'll extract the values 
    for var_index, var in enumerate(variables):
        if var != 'P':
            # if we have (time, lev, lat, lon), then use level parameter
            dimensions = dataset.variables[var].dims
            if dimensions == ('time', 'lev', 'lat', 'lon'):
                values = dataset[var].values[:, lev, :, :]
            elif dimensions == ('time', 'lat', 'lon'):
                values = dataset[var].values[:, :, :]
            else:
                raise ValueError("Unsupported variable dimensions: {dims}".format(dims=dimensions))
        else:
            if nf==1:
                values = (dataset['hyam'].values[lev] * dataset['P0'].values + 
                          dataset['hybm'].values[lev] * dataset['PS'].values[:,:,:]) 
            else:
                values = (dataset['hyam'].values[0,lev] * dataset['P0'].values[0] + 
                          dataset['hybm'].values[0,lev] * dataset['PS'].values[:,:,:]) 
                

            # add the values into the array at the variable's position
        arr[:, :, :, var_index] = values
    
    return arr
    
def extract_datasets(data_dir,training_features,testing_features):    
    netcdf_features_train = []
    netcdf_labels_train = []
    netcdf_features_predict = []
    netcdf_labels_predict = []

    for nf in range(len(training_features)):    
        flnm = data_dir + training_features[nf]
        netcdf_features_train.append(flnm)
        netcdf_labels_train.append(flnm.replace('h0','h1'))

    for nf in range(len(testing_features)):    
        flnm = data_dir + testing_features[nf]
        netcdf_features_predict.append(flnm)
        netcdf_labels_predict.append(flnm.replace('h0','h1'))
        
    features_train = xr.open_mfdataset(netcdf_features_train)
    labels_train = xr.open_mfdataset(netcdf_labels_train)
    features_predict = xr.open_mfdataset(netcdf_features_predict)
    labels_predict = xr.open_mfdataset(netcdf_labels_predict)

    check_datasets(features_train,labels_train)
    return features_train,labels_train,features_predict,labels_predict

def check_datasets(ds1,ds2):
    # confirm that we have datasets that match on the time, lev, lat, and lon dimension/coordinate
    if np.any(ds1.variables['time'].values != ds2.variables['time'].values):
        raise ValueError('Non-matching time values between feature and label datasets')
    if np.any(ds1.variables['lev'].values != ds2.variables['lev'].values):
        raise ValueError('Non-matching level values between feature and label datasets')
    if np.any(ds1.variables['lat'].values != ds2.variables['lat'].values):
        raise ValueError('Non-matching lat values between feature and label datasets')
    if np.any(ds1.variables['lon'].values != ds2.variables['lon'].values):
        raise ValueError('Non-matching lon values between feature and label datasets')

# function to perform scaling
def scale_4d(features_train,
             features_predict,
             labels_train,
             scalers_feature,
             scalers_label,
             labels):
    
    # make new arrays to contain the scaled values we'll return
    scaled_features_train = np.empty(shape=features_train.shape)
    scaled_features_predict = np.empty(shape=features_predict.shape)
    scaled_labels_train = np.empty(shape=labels_train.shape)
    size_times_train = features_train.shape[0]
    size_times_predict = features_predict.shape[0]
    size_lat = features_train.shape[1]
    size_lon = features_train.shape[2]

    
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

