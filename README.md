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

## Project Structure
```
air-quality-forecasting
│
├── configs
│   ├── base.yaml                 # default configuration
│   └── experiments               # configs for LSTM experiments
│       ├── lstm_baseline.yaml
│       ├── lstm_dropout.yaml
│       └── lstm_small_lr.yaml
│
├── data
│   └── raw
│       └── air_quality.csv       # dataset
│
├── notebooks
│   └── analysis.ipynb            # exploratory analysis
│
├── reports
│   ├── experiments               # experiment metrics
│   ├── figures                   # generated plots
│   ├── baselines_metrics.json
│   └── ridge_metrics.json
│
├── src
│   └── air_quality_forecasting
│       ├── baselines.py          # naive forecasting baselines
│       ├── config.py             # config loading utilities
│       ├── datasets.py           # dataset creation and splits
│       ├── evaluate.py           # forecasting metrics
│       ├── generate_plots.py     # generate evaluation plots
│       ├── models.py             # model definitions (Ridge, LSTM)
│       ├── preprocessing.py      # data preprocessing utilities
│       ├── run_baselines.py      # run baseline models
│       ├── run_linear_model.py   # train Ridge regression model
│       ├── run_rnn.py            # train LSTM model
│       ├── utils.py              # helper functions
│       └── visualization.py      # plotting utilities
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Key components

**datasets.py**

Creates train/validation/test splits and constructs sliding window datasets.

**run_baselines.py**

Runs naive forecasting baselines.

**run_linear_model.py**

Trains a Ridge regression model using lagged input windows.

**run_rnn.py**

Trains an LSTM model for multi-step forecasting.

**evaluate.py**

Computes evaluation metrics including MAE, RMSE and R² across forecast horizons.

**visualization.py**

Generates plots for forecast examples and horizon-wise model comparison.