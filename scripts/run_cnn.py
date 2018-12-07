#!/usr/bin/env python

"""
This script illustrates the process of creating simple neural network models using the Keras framework, 
processing data for input to the models for training and prediction, and the use of numpy and xarray 
for data wrangling and I/O with datasets contained within NetCDF files.

To use this script, you must have the Python libraries 'xarray' and 'keras' installed along with all
of their dependencies. You may do this with your own local version of a miniconda install (see install
directions at: https://conda.io/docs/user-guide/install/linux.html), or you may simply add the line: 
    export PATH="/glade/u/home/glimon/miniconda3/bin:$PATH"
to your .bashrc file in your home directory on Cheyenne. 

This script must be run on one of Cheyenne's
Data Analysis and Visualization (DAV) systems, such as Casper. See: 
    https://www2.cisl.ucar.edu/resources/computational-systems/geyser-and-caldera/using-data-analysis-and-visualization-clusters
for details on this.

Lastly, be sure to run:
    chmod +x run_dense.py
on command line before using this script.
"""

import argparse
import numpy as np
import xarray as xr
from keras.models import Sequential
from keras.layers import Dense, Conv3D
from ml_funcs import extrac_data_array, extract_features_labels, scale_4d

parser = argparse.ArgumentParser(description="Script for running Dense Layer Neural Network Machine "+
                                 "Learning model for climate models. Remember to run 'chmod +x run_dense.py'"+
                                 " on command line before using this script.")

parser.add_argument('data_dir', type=str, 
                    help='Path to the directory containing data.')
parser.add_argument('result_dir', type=str, 
                    help='Path to the directory to write our predicted data.')
parser.add_argument('train_features', type=str, nargs='+',
                    help='List of file names of training features data inside data_dir. Separate files with spaces.')
parser.add_argument('train_labels', type=str, nargs='+',
                    help='List of file names of training labels data inside data_dir.')
parser.add_argument('predict_features', type=str, nargs='+',
                    help='List of file names of predicted features data inside data_dir.')
parser.add_argument('predict_labels', type=str, nargs='+',
                    help='List of file names to output  predicted labels data to inside result_dir.')
parser.add_argument('cam_labels', type=str, nargs='+',
                    help='List of file names of CAM output labels of predicted data inside data_dir.')
parser.add_argument('--features', type=str, default=["PS", "T", "U", "V"],
                    help='<help_description>')
parser.add_argument('--labels', type=list, default=["PTTEND"],
                    help='<help_description>')
# parser.add_argument('<arg_name>', type=str,
#                     help='<help_description>')

args = parser.parse_args()

netcdf_features_train = {}
netcdf_labels_train = {}
netcdf_features_predict = {}
netcdf_predict = {}

#Append data_dir and result_dir to appropriate files.
for nf in range(len(args.train_features)):
    netcdf_features_train[nf] = args.data_dir + args.train_features[nf]
    netcdf_labels_train[nf] = args.data_dir + args.train_labels[nf]
for nf in range(len(args.predict_features)):
    args.predict_features[nf] = args.data_dir + args.predict_features[nf]
    netcdf_predict[nf] = (args.result_dir + 
                          args.predict_features[nf][0:len(args.predict_features[nf])-3] + 
                          '_predicted.nc')

# loop over all levels
for lev in range(1,size_lev):
    
    print("Training/predicting for level {level}".format(level=lev))
    
    # get the features and labels for training
    train_x, train_y = extract_features_labels(netcdf_features_train[0],  # Why only the first file??
                                               netcdf_labels_train[0],
                                               features,
                                               labels,
                                               lev)
    
    # get the features for prediction
    predict_x = extract_data_array(xr.open_dataset(netcdf_features_predict[0]),
                                   features,
                                   lev)

    # I added this, James did not seem to have this in the loop and just kept
    # the values from level zero
    size_times_train = train_x.shape[0]
    size_times_predict = predict_x.shape[0]
    size_lat = train_x.shape[1]
    size_lon = train_x.shape[2]
    size_lev = xr.open_dataset(predict_features[0]).lev.size

    # scale the data between 0 and 1
    scalers_x = [MinMaxScaler(feature_range=(0, 1))] * len(features)
    scalers_y = [MinMaxScaler(feature_range=(0, 1))] * len(labels)
    scaled_train_x, scaled_predict_x, scaled_train_y, scalers_x, scalers_y = \
        scale_4d(train_x, predict_x, train_y, scalers_x, scalers_y)

    if (lev == 1) :
        # define the model
        cnn_model = Sequential()

        # add an initial 3-D convolutional layer
        cnn_model.add(Conv3D(filters=32,
                             kernel_size=(3, 3, 3),
                             activation="relu",
                             data_format="channels_last",
                             input_shape=(size_times_train, size_lat, size_lon, len(features)),
                             padding='same'))

        # add a fully-connected hidden layer with twice the number of neurons as input attributes (features)
        cnn_model.add(Dense(len(features) * 2, activation='relu'))

        # output layer uses no activation function since we are interested
        # in predicting numerical values directly without transform
        cnn_model.add(Dense(len(labels)))

        # compile the model using the ADAM optimization algorithm and a mean squared error loss function
        cnn_model.compile(optimizer='adam', loss='mse')

        # display some summary information
        cnn_model.summary()

    # reshape the data for convolutional model input
    shape_x = (1, ) + scaled_train_x.shape
    shape_y = (1, ) + scaled_train_y.shape
    train_x_cnn = np.reshape(scaled_train_x, newshape=shape_x)
    train_y_cnn = np.reshape(scaled_train_y, newshape=shape_y)
    predict_x_cnn = np.reshape(scaled_predict_x, newshape=shape_x)
        
    # train the models
    cnn_model.fit(train_x_cnn, train_y_cnn, shuffle=True, epochs=8, verbose=2)
    
    # use the fitted models to make predictions
    predict_y_scaled_cnn = cnn_model.predict(predict_x_cnn, verbose=1)

    # reverse the scaling of the predicted values
    # TODO below assumes a single label, will need modification for multiple labels
    scaler = scalers_y[0]  # assumes the label scaler was fitted in scale_4d() and side effect carried through
        
    # output from CNN model is 5-D, so we'll flatten first to make it amenable to scaling
    unscaled_predict_y_cnn = scaler.inverse_transform(predict_y_scaled_cnn.flatten().reshape(-1, 1))
    
    # reshape data so it's compatible with assignment into prediction arrays
    level_shape = (size_times_predict, size_lat, size_lon)
    prediction_cnn[:, lev, :, :] = np.reshape(unscaled_predict_y_cnn, newshape=level_shape)

# copy the prediction features dataset since the predicted label(s) should share the same coordinates, etc.
ds_predict_labels = xr.open_dataset(args.data_dir + args.cam_labels)

# remove all non-label data variables from the predictions dataset
for var in ds_predict_labels.data_vars:
    if var not in labels:
        ds_predict_labels = ds_predict_labels.drop(var)

# create new variables to contain the predicted labels, assign these into the prediction dataset
predicted_label_var = xr.Variable(dims=('time', 'lev', 'lat', 'lon'),
                                  data=prediction_cnn,
                                  attrs=ds_predict_labels[labels[0]].attrs)
ds_predict_labels[labels[0] + "_cnn"] = predicted_label_var

# open the dataset containing the computed label values corresponding to the input features used for prediction
ds_cam_labels = xr.open_dataset(data_dir + "/fv091x180L26_moist_HS.cam.h1.2001-02-25-00000_lowres.nc")

# get the differences between computed and predicted values
pttend_diff_cnn = ds_cam_labels[labels[0]].values - prediction_cnn

# create the variables and add to the dataset
predicted_label_var = xr.Variable(dims=('time', 'lev', 'lat', 'lon'),
                                  data=pttend_diff_cnn,
                                  attrs=ds_predict_labels[labels[0]].attrs)
ds_predict_labels[labels[0] + "_cnn_diff"] = predicted_label_var
