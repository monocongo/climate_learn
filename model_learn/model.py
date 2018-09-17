import argparse
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
from sklearn import linear_model, neighbors
from sklearn.model_selection import train_test_split
import xarray as xr

# ----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
def extract_timestamps(ds,
                       initial_year,
                       initial_month,
                       initial_day):

    # Cook up an initial datetime object based on our specified initial date.
    initial = datetime(initial_year, initial_month, initial_day)

    # Create an array of datetime objects from the time values (assumed to be in units of days since the inital date).
    times = ds.variables['time'].values
    datetimes = np.empty(shape=times.shape, dtype='datetime64[m]')
    for i in range(datetimes.size):
        datetimes[i] = initial + timedelta(days=times[i])

    # Put the array into a Series and return it.
    return pd.Series(datetimes)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    This module is used to showcase ML modeling of the climate using scikit-learn, using NCAR CAM files as input.
    """

    try:

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_flows",
                            help="NetCDF file containing flow variables",
                            nargs='*',
                            required=True)
        parser.add_argument("--input_tendencies",
                            help="NetCDF file containing time tendency forcing variables",
                            nargs='*',
                            required=True)
        args = parser.parse_args()

        # open the features (flows) and labels (tendencies) as xarray DataSets
        ds_features = xr.open_mfdataset(args.input_flows)
        ds_labels = xr.open_mfdataset(args.input_tendencies)

        # confirm that we have datasets that match on the time dimension/coordinate
        if (ds_features.variables['time'].values != ds_labels.variables['time'].values).any():
            _logger.info('ERROR: Non-matching time values')
        else:
            _logger.info("OK: time values match as expected")

        # # TODO get initial year, month, and day from the datasets
        # init_year = 2000
        # init_month = 12
        # init_day = 27
        # timestamps = extract_timestamps(ds_features, init_year, init_month, init_day)

        # put features dataset values into a pandas DataFrame
        ps = pd.Series(ds_features.variables['PS'].values[:, :, :].flatten())
        t = pd.Series(ds_features.variables['T'].values[:, 0, :, :].flatten())
        u = pd.Series(ds_features.variables['U'].values[:, 0, :, :].flatten())
        v = pd.Series(ds_features.variables['V'].values[:, 0, :, :].flatten())
        df_features = pd.DataFrame({'PS': ps,
                                    'T': t,
                                    'U': u,
                                    'V': v})
        df_features.index.rename('index', inplace=True)

        # put labels dataset values into a pandas DataFrame
        pttend = pd.Series(ds_labels.variables['PTTEND'].values[:, 0, :, :].flatten())
        putend = pd.Series(ds_labels.variables['PUTEND'].values[:, 0, :, :].flatten())
        pvtend = pd.Series(ds_labels.variables['PVTEND'].values[:, 0, :, :].flatten())
        df_labels = pd.DataFrame({'PTTEND': pttend,
                                  'PUTEND': putend,
                                  'PVTEND': pvtend})
        df_labels.index.rename('index', inplace=True)

        # split the data into training and testing datasets
        x_train, x_test, y_train, y_test = train_test_split(df_features,
                                                            df_labels,
                                                            test_size=0.25,
                                                            random_state=4)

        # create and train a linear regression model
        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        _logger.info("LRM score: {result}".format(result=score))

        # create and train a ridge regression model
        model = linear_model.Ridge()
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        _logger.info("Ridge score: {result}".format(result=score))

        # create and train a K-neighbors regression model
        for k in [1, 3, 5, 10, 20]:
            model = neighbors.KNeighborsRegressor(n_neighbors = 5)
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            _logger.info("K-Neighbors (k={k}) score: {result}".format(k=k, result=score))

    except Exception as ex:

        _logger.exception('Failed to complete', exc_info=True)
        raise
