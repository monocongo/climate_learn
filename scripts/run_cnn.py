#!/usr/bin/env python

import argparse
import numpy as np
import xarray as xr
from keras.models import Sequential
from keras.layers import Dense, Conv3D
from ml_funcs import extract_data_array, extract_features_labels, scale_4d

parser = argparse.ArgumentParser(description= """ \n
This script illustrates the process of creating simple neural network models using the Keras framework, 
processing data for input to the models for training and prediction, and the use of numpy and xarray 
for data wrangling and I/O with datasets contained within NetCDF files. 

To use this script, you must have the Python libraries 'xarray' and 'keras' installed along with all 
of their dependencies. You may do this with your own local version of a miniconda install (see install 
directions at: https://conda.io/docs/user-guide/install/linux.html), or you may simply add the line: 
    export PATH="/glade/u/home/glimon/miniconda3/bin:$PATH" 
to your .bashrc file in your home directory on Cheyenne. 
This script is meant to be run on one of Cheyenne's Data Analysis and Visualization (DAV) systems, such as Casper. 
See: 
    https://www2.cisl.ucar.edu/resources/computational-systems/geyser-and-caldera/using-data-analysis-and-visualization-clusters 
for details on this. 

Lastly, to run this script as an executable, first run the command: \n
    chmod +x run_cnn.py 
then run the script with: 
    ./run_cnn.py [-args]
otherwise, you may run with:
    python run_cnn.py [-args] 

For documentation of the Keras library used in this scrips to perform the machine learning and 
develop the neural networks, see: 
    https://keras.io/ 

Example usage of this script would be: 
    ./run_cnn.py <path2DataDirectory> <path2ResultDirectory> "<trainingFeaturesFileName_1> \ 
    <trainingFeaturesFileName_2> <...>" "<testingFeaturesFileName_1> <testingFeaturesFileName_2> <...>" \ 
    --<optional_arg_1_flag> <optional_arg_1> --<optional_arg_2_flag> <optional_arg_2> <...> 
""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('data_dir', type=str, 
                    help='Path to the directory containing data.')
parser.add_argument('result_dir', type=str, 
                    help='Path to the directory to write our predicted data.')
parser.add_argument('training_features', type=str, nargs='+', 
                    help='List of file names of training features data inside data_dir. Corresponding to .h0. '+
                    '<name> files. Separate files with spaces.')
parser.add_argument('testing_features', type=str, nargs='+', 
                    help='List of file names of predicted features data inside data_dir. Corresponding to .h0. ' +
                    ' <name> files. Separate files with spaces.')
parser.add_argument('--features', type=str, nargs='+', default="PS T U V",
                    help='Choices of "features." Dynamics variables are chosen as default and used to predict '+
                    'physical tendancies.')
parser.add_argument('--labels', type=list, nargs='+', default="PTTEND",
                    help='Choices of "labels." Physical tendancies that are to be predicted from the '+
                    'dynamics variables.')
parser.add_argument('--activation_function', type=str, default="relu",
                    help='Neural Net activation function to be used. See Keras documentation to see alternate '+
                    'choices.')
parser.add_argument('--optimizer', type=str, default="adam",
                    help='Neural Net optimizer to be used. See Keras documentation to see alternate '+
                    'choices.')
parser.add_argument('--loss_function', type=str, default="mse",
                    help='Neural Net loss function to be used. See Keras documentation to see alternate '+
                    'choices.')
parser.add_argument('--filters', type=int, default=32,
                    help='See Keras documentation to see alternate choices.')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='See Keras documentation to see alternate choices.')
parser.add_argument('--data_format', type=str, default="channels_last",
                    help='See Keras documentation to see alternate choices.')
parser.add_argument('--padding', type=str, default="same",
                    help='See Keras documentation to see alternate choices.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs (iterations) to use when "fitting" Neural Net. See Keras +' 
                    'documentation to see alternate choices.')
parser.add_argument('--shuffle', type=str, default="true",
                    help='Whether or not to shuffle the data in the neural net, for most cases, this should be '+
                    'lefts as the default "true"')
parser.add_argument('--verbose_fit', type=int, default=0,
                    help='Verbosity during the fitting.')
parser.add_argument('--verbose_predict', type=int, default=0,
                    help='Neural Net activation function to be used. See Keras documentation to see alternate '+
                    'choices.')

args = parser.parse_args()

if (args.shuffle=='false'):
    shuff = False
else:
    shuff = True

netcdf_features_train = {}
netcdf_labels_train = {}
netcdf_features_predict = {}
netcdf_labels_predicted = {}
netcdf_labels_cam = {}

#Append data_dir and result_dir to appropriate files.
for nf in range(len(args.train_features)):
    netcdf_features_train[nf] = args.data_dir + args.training_features[nf]
    netcdf_labels_train[nf] = netcdf_features_train[nf].replace('.h0.','.h1.')
for nf in range(len(args.testing_features)):
    flnm = args.testing_features[nf]
    netcdf_features_predict[nf] = args.data_dir + flm
    netcdf_labels_cam[nf] = args.data_dir + flm.replace('.h0.','.h1.')
    netcdf_labels_predicted[nf] = args.result_dir + flm.replace('.h0.','.h1.')[0:length(flnm)-3]+'_predicted.nc'

# loop over all levels
for lev in range(1,size_lev):
    
    print("Training/predicting for level {level}".format(level=lev))
    
    # get the features and labels for training
    train_x, train_y = extract_features_labels(netcdf_features_train[0], #Zero right now because only one file
                                               netcdf_labels_train[0], #Zero right now because only one file
                                               features,
                                               labels,
                                               lev)
    
    # get the features for prediction
    predict_x = extract_data_array(xr.open_dataset(netcdf_features_predict[0]), #Zero right now because only one file
                                   features,
                                   lev)

    # I added this, James did not seem to have this in the loop and just kept
    # the values from level zero
    size_times_train = train_x.shape[0]
    size_times_predict = predict_x.shape[0]
    size_lat = train_x.shape[1]
    size_lon = train_x.shape[2]
    size_lev = xr.open_dataset(predict_features[0]).lev.size #Zero right now because only one file

    # scale the data between 0 and 1
    scalers_x = [MinMaxScaler(feature_range=(0, 1))] * len(features)
    scalers_y = [MinMaxScaler(feature_range=(0, 1))] * len(labels)
    scaled_train_x, scaled_predict_x, scaled_train_y, scalers_x, scalers_y = \
        scale_4d(train_x, predict_x, train_y, scalers_x, scalers_y)
    
    if (lev == 1) : # create the model only once.
        # define the model
        model = Sequential()

        # add an initial 3-D convolutional layer
        model.add(Conv3D(filters=args.filters,
                         kernel_size=kernelSize,
                         activation=args.activation_function,
                         data_format=args.data_format,
                         input_shape=(size_times_train, size_lat, 
                                      size_lon, len(features)),
                         padding=args.padding))

        # add a fully-connected hidden layer with twice the number of neurons as input attributes (features)
        model.add(Dense(len(features) * 2, activation=args.activation_function))
        
        # output layer uses no activation function since we are interested
        # in predicting numerical values directly without transform
        model.add(Dense(len(labels)))
        
        # compile the model using the ADAM optimization algorithm and a mean squared error loss function
        model.compile(optimizer=args.optimizer, loss=args.loss_function)

        # display some summary information
        model.summary()


    # reshape the data for cnn layer model input
    shape_x = (1, ) + scaled_train_x.shape
    shape_y = (1, ) + scaled_train_y.shape
    train_x = np.reshape(scaled_train_x, newshape=shape_x)
    train_y = np.reshape(scaled_train_y, newshape=shape_y)
    predict_x = np.reshape(scaled_predict_x, newshape=shape_x)
    
    # train the models
    model.fit(train_x, train_y, shuffle=shuff, epochs=int(args.epochs), verbose=args.verbose_fit)
    
    # use the fitted models to make predictions
    predict_y_scaled = model.predict(predict_x, verbose=args.verbose_predict)

    # reverse the scaling of the predicted values
    # TODO below assumes a single label, will need modification for multiple labels
    scaler = scalers_y[0]  # assumes the label scaler was fitted in scale_4d() and side effect carried through

    # output from the dense model is 2-D, good for scaler input
    unscaled_predict_y = scaler.inverse_transform(predict_y_scaled)
    
    # reshape data so it's compatible with assignment into prediction arrays
    level_shape = (size_times_predict, size_lat, size_lon)
    prediction[:, lev, :, :] = np.reshape(unscaled_predict_y, newshape=level_shape)

# need to figure out what is happening under here.

#copy the prediction features dataset since the predicted label(s) should share the same coordinates, etc.
ds_predict_labels = xr.open_dataset(netcdf_features_predict[0]) #Zero right now because only one file

# remove all non-label data variables from the predictions dataset
for var in ds_predict_labels.data_vars:
    if var not in labels:
        ds_predict_labels = ds_predict_labels.drop(var)

# create new variables to contain the predicted labels, assign these into the prediction dataset
predicted_label_var = xr.Variable(dims=('time', 'lev', 'lat', 'lon'),
                                  data=prediction,
                                  attrs=ds_predict_labels[labels[0]].attrs)
ds_predict_labels[labels[0]] = predicted_label_var

# open the dataset containing the computed label values corresponding to the input features used for prediction
ds_cam_labels = xr.open_dataset(netcdf_labels_cam[0]) # This is hard coded to the zeroth because current
                                                       # implementation only has one file, need to revisit this.

# write the predicted label(s)' dataset as a NetCDF file
ds_predict_labels.to_netcdf(netcdf_labels_predicted[0]) #Zero right now because only one file
