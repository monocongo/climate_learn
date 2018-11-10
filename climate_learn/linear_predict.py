import argparse
from datetime import datetime
import logging
import math

from matplotlib import pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import sklearn train_test_split
from sklearn import linear_model, metrics

#-------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as stderr
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
def _extract_train_test(ds_features,
                        ds_targets,
                        target_var,
                        level,
                        test_ratio=0.25):
    
    # put feature variables into a Pandas DataFrame
    ps = pd.Series(ds_features['PS'][:, :, :].flatten())
    t = pd.Series(ds_features['T'][:, level, :, :].flatten())
    u = pd.Series(ds_features['U'][:, level, :, :].flatten())
    v = pd.Series(ds_features['V'][:, level, :, :].flatten())
    df_features = pd.DataFrame({'PS': ps,
                                'T': t,
                                'U': u,
                                'V': v})

    # put target variable into a Pandas DataFrame
    forcing_target = pd.Series(ds_targets[target_var][:, level, :, :].flatten())
    df_targets = pd.DataFrame({target_var: forcing_target})

    # split into training and test datasets
    return sklearn.model_selection.train_test_split(df_features,
                                                    df_targets,
                                                    test_size=test_ratio,
                                                    random_state=4)
    
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This module is used to test machine learning performance on climate model
    data from NCAR CAM runs. Target NetCDF datasets are recreated from
    predictions after training/validating from CAM input/output files.
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        _logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--netcdf_features",
                            help="NetCDF file containing flow variables",
                            required=True)
        parser.add_argument("--netcdf_labels",
                            help="NetCDF file containing forcing variables",
                            required=True)
        parser.add_argument("--netcdf_predictions",
                            help="NetCDF file containing predicted forcings",
                            required=True)
        parser.add_argument('--plot', 
                            help='Plot RMSE results per elevation level',
                            action='store_true')
        args = parser.parse_args()

        # Copy the NetCDF containing labels into a NetCDF with corresponding coordinates, variables, and attributes.
        with netCDF4.Dataset(args.netcdf_labels, 'r') as ds_labels, \
            netCDF4.Dataset(args.netcdf_predictions, 'w') as ds_predictions:

            for vars

        # create a linear regression model
        linear_regressor = linear_model.LinearRegression()

        # lists of RMSE values we'll use for plotting
        rmse_training = []        
        rmse_testing = []        

        # open the NetCDFs within a context manager
        with netCDF4.Dataset(args.netcdf_features) as ds_features, \
            netCDF4.Dataset(args.netcdf_labels) as ds_targets, \


            # loop over all elevations, in order to keep memory footprint low
            for lev in range(len(ds_features['lev'])):
        
                _logger.info("Processing for elevation {}".format(lev))
                
                # for each target variable we'll train/test our model, then
                # use the trained model to predict output (target) values
                for target in ['PTTEND', 'PUTEND', 'PVTEND']:

                    # get training and testing features and targets datasets
                    x_train, x_test, y_train, y_test = \
                        _extract_train_test(ds_features,
                                            ds_targets,
                                            target,
                                            lev,
                                            test_ratio=0.1)
                    
                    # train the model
                    linear_regressor = linear_regressor.fit(x_train, y_train)
                    
                    if args.netcdf_predict is not None:
                        with netCDF4.Dataset(args.netcdf_predict) as ds_predict:

                            # predict from the same feature variables used
                            # in the training datasets, (PS, T, U, and V)
                            # at the same level
                            target_predictions = linear_regressor.predict(
                                    ds_features['PS'][:, :, :].flatten(),
                                    ds_features['T'][:, lev, :, :].flatten(),
                                    ds_features['U'][:, lev, :, :].flatten(),
                                    ds_features['V'][:, lev, :, :].flatten())

                            # reshape accordingly
                            new_shape = ds_targets[target].shape
                            new_shape[1] = 1   # 2nd dimension is lev, we have 1
                            target_predictions = np.reshape(target_predictions,
                                                            new_shape)

                    # if plotting we'll need to retain the metrics
                    if args.plot:
                        
                        # use the trained model to make predictions
                        predictions_train = linear_regressor.predict(x_train)
                        predictions_test = linear_regressor.predict(x_test)
        
                        # compute training and validation loss
                        rmse_train = math.sqrt(
                            metrics.mean_squared_error(predictions_train, y_train))
                        rmse_test = math.sqrt(
                            metrics.mean_squared_error(predictions_test, y_test))
    
                        # Add the loss metrics from this period to our lists.
                        rmse_training.append(rmse_train)
                        rmse_testing.append(rmse_test)
                    
                    _logger.info("  RMSE (train): {}".format(rmse_train))
                    _logger.info("  RMSE (test):  {}".format(rmse_test))
               
        # Output a graph of loss metrics over elevations.
        if args.plot:
            plt.ylabel("RMSE")
            plt.xlabel("Elevation")
            plt.title("Root Mean Squared Error vs. Elevations")
            plt.tight_layout()
            plt.plot(rmse_training, label="training")
            plt.plot(rmse_testing, label="validation")
            plt.legend()
            plt.show()
        
        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
