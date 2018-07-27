import argparse
from datetime import datetime
import logging
import math

from matplotlib import pyplot as plt
import netCDF4
import pandas as pd
from sklearn.model_selection import train_test_split
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
                        lev,
                        test_ratio=0.25):
    
    # put feature variables into a Pandas DataFrame
    ps = pd.Series(ds_features['PS'][:, :, :].flatten())
    t = pd.Series(ds_features['T'][:, lev, :, :].flatten())
    u = pd.Series(ds_features['U'][:, lev, :, :].flatten())
    v = pd.Series(ds_features['V'][:, lev, :, :].flatten())
    df_features = pd.DataFrame({'PS': ps,
                                'T': t,
                                'U': u,
                                'V': v})

    # put target variables into a Pandas DataFrame
    pttend = pd.Series(ds_targets['PTTEND'][:, lev, :, :].flatten())
    putend = pd.Series(ds_targets['PUTEND'][:, lev, :, :].flatten())
    pvtend = pd.Series(ds_targets['PVTEND'][:, lev, :, :].flatten())
    df_targets = pd.DataFrame({'PTTEND': pttend,
                               'PUTEND': putend,
                               'PVTEND': pvtend})
        
    # split into training and test datasets
    return train_test_split(df_features,
                            df_targets, 
                            test_size=test_ratio, 
                            random_state=4)
    
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This module is used to test machine learning performance on climate model
    data from NCAR CAM runs.
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
                            help="NetCDF file containing forcing variables",
                            required=True)
        args = parser.parse_args()

        # create a linear regression model
        linear_regressor = linear_model.LinearRegression()

        # lists of RMSE values we'll use for plotting
        rmse_training = []        
        rmse_testing = []        

        # open the NetCDFs within a context manager
        with netCDF4.Dataset(args.netcdf_flows) as ds_features, \
            netCDF4.Dataset(args.netcdf_tendencies) as ds_targets:

            # loop over all elevations, in order to keep memory footprint low
            for lev in range(len(ds_features['lev'])):
        
                _logger.info("Processing for elevation {}".format(lev))
                
                # get training and testing features and targets datasets
                x_train, x_test, y_train, y_test = \
                    _extract_train_test(ds_features,
                                        ds_targets,
                                        lev)
                
                # train the model
                linear_regressor = linear_regressor.fit(x_train, y_train)
                
                # use the trained model to make predictions
                predictions_train = linear_regressor.predict(x_train)
                predictions_test = linear_regressor.predict(x_test)

                # compute training and validation loss
                rmse_train = math.sqrt(
                    metrics.mean_squared_error(predictions_train, y_train))
                rmse_test = math.sqrt(
                    metrics.mean_squared_error(predictions_test, y_test))

                # Add the loss metrics from this period to our list.
                rmse_training.append(rmse_train)
                rmse_testing.append(rmse_test)
                
                _logger.info("  RMSE (train): {}".format(rmse_train))
                _logger.info("  RMSE (test):  {}".format(rmse_test))
               
        # Output a graph of loss metrics over elevations.
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
