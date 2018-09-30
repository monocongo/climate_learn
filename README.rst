=======================================
Climate Modeling using Machine Learning
=======================================

Research into using a machine learning (ML) approach for climate modeling.
--------------------------------------------------------------------------
Using the results of the `NCAR Community Atmosphere Model (CAM) <http://www.cesm.ucar.edu/models/atm-cam/>`_ as a basis
we attempt to create a ML model that matches CAM results in order to
demonstrate suitability. The model will then be tuned for various climate scenarios, including
the prediction of temperature and precipitation matching observational datasets.

Methodology
-----------

- Using `Keras <https://keras.io>`_/`TensorFlow <https://www.tensorflow.org/>`_ we develop a ML-based climate model that accurately predicts the results produced by CAM, using CAM inputs/outputs for training and evaluation, in order to approximately duplicate the capabilities of that model.
- We refine the ML-based model for observational inputs/outputs, for use in later comparisons of model results vs. observed climatologies.

