=======================================
Climate Modeling using Machine Learning
=======================================

Research into using a machine learning (ML) approach for climate modeling
--------------------------------------------------------------------------
Using the results of the `NCAR Community Atmosphere Model (CAM) <http://www.cesm.ucar.edu/models/atm-cam/>`_ as a basis
we attempt to create a ML model that matches CAM results in order to
demonstrate suitability. The model will then be tuned for various climate scenarios, including
the prediction of temperature and precipitation matching observational datasets.

Goals
-----------

- Develop a ML-based climate model using `Keras <https://keras.io>`_/`TensorFlow <https://www.tensorflow.org/>`_ which accurately predicts the results produced by CAM, using CAM inputs/outputs for training and evaluation, in order to approximately duplicate the capabilities of that model.
- Refine the ML-based model for observational inputs/outputs, for use in later comparisons of model results vs. observed climatologies.
- Explore the extensibility of the model for predicting climate monitoring indices such as `SPI <https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi>`_, `SPEI <https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-evapotranspiration-index-spei>`_, and `PDSI <https://climatedataguide.ucar.edu/climate-data/palmer-drought-severity-index-pdsi>`_.

Relevant literature
-------------------
`Convolutional LSTM Network: A Machine Learning
Approach for Precipitation Nowcasting (Shi, Chen, Wang, Yeung) <https://arxiv.org/pdf/1506.04214.pdf>`_

`Machine learning of atmospheric chemistry (Evans, Keller) <http://adsabs.harvard.edu/abs/2017AGUFM.A41H2384E>`_

`Downscaling of precipitation for climate change scenarios: A support vector machine approach (Tripathi, Srinivas, Nanjundiah) <https://doi.org/10.1016/j.jhydrol.2006.04.030>`_

`Multi-stage committee based extreme learning machine model incorporating the influence of climate parameters and seasonality on drought forecasting (Ali, Deo, Downs, Maraseni) <https://doi.org/10.1016/j.compag.2018.07.013>`_

`An extensive evaluation of seven machine learning methods for rainfall prediction in weather derivatives (Cramer, Kampouridis, Freitas, Alexandridis) <https://www.sciencedirect.com/science/article/pii/S0957417417303457>`_

`Machine learning for ecosystem services (Willcock, et al) <https://www.sciencedirect.com/science/article/pii/S2212041617306423>`_

`Meteorological drought forecasting for ungauged areas based on machine learning: Using long-range climate forecast and remote sensing data (Rhee, Im) <https://www.sciencedirect.com/science/article/pii/S0168192317300448>`_

`Meteorological Drought Forecasting Based on Climate Signals Using Artificial Neural Network – A Case Study in Khanhhoa Province Vietnam (Le, Perez, Solomatine, Nguyen) <https://www.sciencedirect.com/science/article/pii/S1877705816319178>`_

`Drought assessment and monitoring through blending of multi-sensor indices using machine learning approaches for different climate regions (Park, Im, Jang, Rhee) <https://doi.org/10.1016/j.agrformet.2015.10.011>`_

`Neural network modelling for the analysis of forcings/temperatures relationships at different scales in the climate system (Pasini, Lore, Ameli) <https://doi.org/10.1016/j.ecolmodel.2005.08.012>`_

`The application of machine learning for evaluating anthropogenic versus natural climate change (Abbot, Marohasy) <https://doi.org/10.1016/j.grj.2017.08.001>`_