import argparse
from datetime import datetime
import logging
import math

# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import xarray as xr

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.2f}'.format

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
def get_input(features_ds, 
              targets_ds, 
              batch_size=1, 
              shuffle=True, 
              num_epochs=None):
    """
    Extracts a batch of elements from a dataset.
  
    Args:
      features: xarray Dataset of features, with n feature variables
      targets: xarray Dataset of targets, with one target variable
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. 
                  None == repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Get the shape of the initial feature variable, which we assume is the same
    # as all other feature and target variables. The we'll use the product of 
    # the final three values (lev, lat, and lon) as the first shape value we'll
    # use for our reshaped feature arrays, and use the variable's first shape
    # value as the second shape value of our reshaped feature arrays.
    var_shape = list(features_ds.variables.items())[0][1].shape
    s0 = var_shape[1] * var_shape[2] * var_shape[3]
    s1 = var_shape[0]
    
    # Convert xarray data into a dict of numpy arrays.
    # For each feature variable we have a 4-D array of values, with dimensions
    # (time, lev, lat, lon). We'll swap the axes to (lon, lat, lev, time) then
    # reshape the array to 2-D with shape (lon*lat*lev, time), i.e. (s0, s1).
    features = {}
    for var in features_ds.variables:
        v = features_ds[var]  # the variable itself is an xarray.DataArray
        features[var] = v.values.swapaxes(0, 3).swapaxes(1, 2).reshape(s0, s1)
        
    # Perform the same for the target dataset, which we assume contains a single
    # variable with the same initial shape as the feature dataset's variables.
    targets = {}
    var = targets_ds.variables[0]
    targets[var] = var.values.swapaxes(0, 3).swapaxes(1, 2).reshape(s0, s1)
    
    # Construct a TensorFlow Dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This module is used to perform climate indices processing on gridded datasets in NetCDF.
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        _logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--netcdf_flows",
                            help="NetCDF file containing flow variables",
                            required=True)
        parser.add_argument("--netcdf_tendencies",
                            help="NetCDF file containing time tendency forcing variables",
                            required=True)
        parser.add_argument("--layers",
                            help="Number of nodes per layer",
                            type=int,
                            nargs = '*')
        args = parser.parse_args()

        
        # Load the flow and time tendency forcing datasets into Xarray Dataset objects.
        data_h0 = xr.open_dataset(args.netcdf_flows, decode_times=False)
        data_h1 = xr.open_dataset(args.netcdf_tendencies, decode_times=False)
        
        """
        ## Define the features and configure feature columns
        
        In TensorFlow, we indicate a feature's data type using a construct
        called a feature column. Feature columns store only a description 
        of the feature data; they do not contain the feature data itself.
        As features we'll use the following flow variables:
        
        * U (west-east (zonal) wind, m/s)
        * V (south-north (meridional) wind, m/s)
        * T (temperature, K)
        * PS (surface pressure, Pa)
        
        We'll take the flow variables dataset and trim out all but the
        above variables, and use this as the data source for features.
        
        The variables correspond to Numpy arrays, and we'll use the shapes of
        the variable arrays as the shapes of the corresponding feature columns.
        """
        
        # Define the input features as PS, T, U, and V.
        
        # Remove all non-feature variables from the dataset.
        feature_vars = ['T', 'U', 'V']
        # include PS once we understand how to deal with a variable that
        # has different dimensions, since PS is missing the lev dimension
        #feature_vars = ['PS', 'T', 'U', 'V']
        for var in data_h0.variables:
            if var not in feature_vars:
                data_h0 = data_h0.drop(var)
        
        # Configure numeric feature columns for the input features.
        # Each column will contain a 1-D array representing the time series
        # for a geospatial point (x, y, z) or (lon, lat, lev), so the shape 
        # of the column will be the number of time steps, i.e. the size of the time
        # dimension of the variable. The variables are assumed to have dimensions
        # (time, lev, lat, lon).
        feature_columns = []
        for var in feature_vars:
            feature_columns.append(
                tf.feature_column.numeric_column(var, shape=data_h0[var].shape[0]))
            
        """
        ## Define the targets (labels)
        
        Time tendency forcings are the targets (labels) that our model
        should learn to predict.
        
        * PTTEND (time tendency of the temperature)
        * PUTEND (time tendency of the zonal wind)
        * PVTEND (time tendency of the meridional wind)
        
        We'll take the time tendency forcings dataset and trim out all
        other variables so we can use this as the data source for targets.
        """
        
        # Define the targets (labels) as PTTEND, PUTEND, and PVTEND.
        # NOTE only using a single target variable for now until we can work out 
        # how to instead use all three at once (perhaps as a list or 2-D array, 
        # i.e. [PTTEND_timeseries, PUTEND_timeseries, PVTEND_timeseries])
        
        # Remove all non-target variables dataset.
        target_vars = ['PTTEND']
        #target_vars = ['PTTEND', 'PUTEND', 'PVTEND']
        for var in data_h1.variables:
            if var not in target_vars:
                data_h1 = data_h1.drop(var)

        # Confirm the compatability of our features and targets datasets,
        # in terms of dimensions and coordinates.
        if data_h0.dims != data_h1.dims:
            print("WARNING: Unequal dimensions")
        else:
            for var_h0 in data_h0.variables:
                for var_h1 in data_h1.variables:
                    if data_h0[var_h0].values.shape != data_h1[var_h1].values.shape:
                        print("WARNING: Unequal shapes for feature {} and target {}".format(var_h0, var_h1))
        
        """
        ## Split the data into training, validation, and testing datasets
        
        We'll initially split the dataset into training, validation, 
        and testing datasets with 50% for training and 25% each for 
        validation and testing. We'll use the longitude dimension 
        to split since it has 180 points and divides evenly by four.
        We get every other longitude starting at the first longitude to get
        50% of the dataset for training, then every fourth longitude
        starting at the second longitude to get 25% of the dataset for 
        validation, and every fourth longitude starting at the fourth
        longitude to get 25% of the dataset for testing.
        """        
        lon_range_training = list(range(0, data_h0.dims['lon'], 2))
        lon_range_validation = list(range(1, data_h0.dims['lon'], 4))
        lon_range_testing = list(range(3, data_h0.dims['lon'], 4))
        
        features_training = data_h0.isel(lon=lon_range_training)
        features_validation = data_h0.isel(lon=lon_range_validation)
        features_testing = data_h0.isel(lon=lon_range_testing)
        
        targets_training = data_h1.isel(lon=lon_range_training)
        targets_validation = data_h1.isel(lon=lon_range_validation)
        targets_testing = data_h1.isel(lon=lon_range_testing)
        
        """
        ## Create the neural network
        
        Next, we'll instantiate and configure a neural network using
        TensorFlow's [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor)
        class. We'll train this model using the GradientDescentOptimizer,
        which implements Mini-Batch Stochastic Gradient Descent (SGD).
        The learning_rate argument controls the size of the gradient step.
        
        NOTE: To be safe, we also apply gradient clipping to our optimizer via
        `clip_gradients_by_norm`. Gradient clipping ensures the magnitude of
        the gradients do not become too large during training, which can cause
        gradient descent to fail.
        
        We use `hidden_units`to define the structure of the NN.
        The `hidden_units` argument provides a list of ints, where each int
        corresponds to a hidden layer and indicates the number of nodes in it.
        For example, consider the following assignment:
        
        `hidden_units=[3, 10]`
        
        The preceding assignment specifies a neural net with two hidden layers:
        
        The first hidden layer contains 3 nodes.
        The second hidden layer contains 10 nodes.
        If we wanted to add more layers, we'd add more ints to the list.
        For example, `hidden_units=[10, 20, 30, 40]` would create four layers
        with ten, twenty, thirty, and forty units, respectively.
        
        By default, all hidden layers will use ReLu activation and will be fully connected.
        """
        
        # Use gradient descent as the optimizer for training the model.
        gd_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        gd_optimizer = tf.contrib.estimator.clip_gradients_by_norm(gd_optimizer, 5.0)
        
        # Use hidden layers with the number of nodes specified as command arguments.
        hidden_units = args.layers
        
        # Instantiate the neural network.
        dnn_regressor = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                                  hidden_units=hidden_units,
                                                  optimizer=gd_optimizer)
        
        # Create input functions. Wrap get_input() in a lambda so we 
        # can pass in features and targets as arguments.
        input_training = lambda: get_input(features_training, 
                                           targets_training, 
                                           batch_size=10)
        predict_input_training = lambda: get_input(features_training, 
                                                   targets_training, 
                                                   num_epochs=1, 
                                                   shuffle=False)
        predict_input_validation = lambda: get_input(features_validation, 
                                                     targets_validation, 
                                                     num_epochs=1, 
                                                     shuffle=False)
        
        """## Train and evaluate the model
        
        We can now call `train()` on our `dnn_regressor` to train the model. 
        We'll loop over a number of periods and on each loop we'll train
        the model, use it to make predictions, and compute the RMSE of the
        loss for both training and validation datasets.
        """
        
        print("Training model...")
        print("RMSE (on training data):")
        training_rmse = []
        validation_rmse = []
        
        steps = 500
        periods = 20
        steps_per_period = steps / periods
        
        # Train the model inside a loop so that we can periodically assess loss metrics.
        for period in range (0, periods):
        
            # Train the model, starting from the prior state.
            dnn_regressor.train(input_fn=input_training,
                                steps=steps_per_period)
        
            # Take a break and compute predictions, converting to numpy arrays.
            training_predictions = dnn_regressor.predict(input_fn=predict_input_training)
            training_predictions = np.array([item['predictions'][0] for item in training_predictions])
            
            validation_predictions = dnn_regressor.predict(input_fn=predict_input_validation)
            validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
            
            # Compute training and validation loss.
            rmse_train = math.sqrt(metrics.mean_squared_error(training_predictions, 
                                                              targets_training))
            rmse_valid = math.sqrt(metrics.mean_squared_error(validation_predictions, 
                                                              targets_validation))
            
            # Print the current loss.
            print("  period %02d : %0.2f" % (period, rmse_train))
            
            # Add the loss metrics from this period to our list.
            training_rmse.append(rmse_train)
            validation_rmse.append(rmse_valid)
        
        print("Model training finished.")
        
#         # Output a graph of loss metrics over periods.
#         plt.ylabel("RMSE")
#         plt.xlabel("Periods")
#         plt.title("Root Mean Squared Error vs. Periods")
#         plt.tight_layout()
#         plt.plot(rmse_train, label="training")
#         plt.plot(validation_rmse, label="validation")
#         plt.legend()
        
        print("Final RMSE (on training data):   %0.2f" % rmse_train)
        print("Final RMSE (on validation data): %0.2f" % rmse_valid)

        # Create an input function for test dataset.
        predict_input_testing = lambda: get_input(features_testing, 
                                                  targets_testing, 
                                                  num_epochs=1, 
                                                  shuffle=False)
        
        # Get predictions for testing dataset.
        predictions_testing = dnn_regressor.predict(input_fn=predict_input_testing)
        predictions_testing = np.array([item['predictions'][0] for item in predictions_testing])
            
        # Compute training and validation loss.
        rmse_test = math.sqrt(metrics.mean_squared_error(predictions_testing, targets_testing))
        
        # Print the current loss.
        print("Testing RMSE: %0.2f" % (rmse_test))

        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
