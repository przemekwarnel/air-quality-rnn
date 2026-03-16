# Air Quality Forecasting with Machine Learning and LSTM

## Project Overview 

This project builds and evaluates machine learning models for **multi-step time series forecasting of air pollution levels**.

Using historical air quality and meteorological data, the goal is to predict **PM2.5 concentration for the next 24 hours** based on the previous 48 hours of observations.

The project compares:

- naive time-series baselines
- regularized linear regression (Ridge)
- a deep learning model based on **LSTM**

All models are evaluated using **multi-horizon forecasting metrics** and compared across forecast horizons.

The repository demonstrates a full **Applied ML workflow for time-series forecasting**, including:

- data preprocessing
- feature scaling
- sliding window dataset construction
- baseline modelling
- neural network training
- horizon-wise evaluation
- visualization of forecast performance

## Dataset

The project uses an air quality dataset containing hourly observations of pollution and meteorological variables.

The dataset includes measurements such as:

- PM2.5 concentration (target variable)
- temperature
- pressure
- dew point
- wind speed
- precipitation
- other atmospheric indicators

Each row corresponds to **one hour of observations**.

The dataset is included in the repository under:

[`data/raw/PRSA_Data_Wanshouxigong_20130301-20170228.csv`](data/raw/PRSA_Data_Wanshouxigong_20130301-20170228.csv)

### Target

The forecasting target is:

```
PM2.5 concentration
```

### Forecasting task

The models predict **24 future hourly values** of PM2.5 based on the **previous 48 hours of data**.

```
Input window: 48 hours
Forecast horizon: 24 hours
```

## Problem Formulation

The forecasting task is formulated as a **multi-step time series regression problem**.

For each training example:
```
X = past 48 hours of observations
y = next 24 hours of PM2.5 values
```

The dataset is converted into supervised learning samples using a **sliding window approach**.


Evaluation is performed using a **chronological train / validation / test split** to avoid data leakage.

Metrics are computed for:

- the entire forecast horizon
- each individual forecast step