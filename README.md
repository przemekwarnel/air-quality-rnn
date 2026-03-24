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

## Key Features 

- Chronological time-series split
- Sliding window dataset construction
- Multi-horizon forecasting (24 steps)
- Comparison of classical ML and deep learning models
- Horizon-wise evaluation metrics
  
## Tech Stack

- **Python**
- **pandas / NumPy** — data processing
- **scikit-learn** — classical machine learning
- **TensorFlow / Keras** — deep learning (LSTM)
- **Matplotlib** — visualization
- **YAML** — configuration management

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

## Methodology

### Data preprocessing

The dataset consists of hourly air quality and meteorological observations.  
Data preprocessing includes:

- chronological sorting of observations
- handling missing values via imputation
- feature scaling using `StandardScaler`
- separate scaling of features and target variable

Scaling is performed **after the train/validation/test split** to avoid data leakage.

### Sliding window dataset

The time series is transformed into a supervised learning dataset using a **sliding window approach**.

Each training example consists of:

```
Input: previous 48 hours of observations
Target: next 24 hours of PM2.5 values
```

This produces input tensors with shape:

```
(samples, window_size, n_features)
```

and targets:

```
(samples, forecast_horizon)
```

### Models

The project compares several forecasting approaches:

**Naive baseline**

Uses the most recent observation as the prediction for all future time steps.

**Seasonal naive baseline**

Uses the observation from the same hour in the previous daily cycle as the forecast.
This baseline captures simple daily seasonality in the time series.

**Ridge regression**

A regularized linear regression model trained on flattened input windows.

This model serves as a strong classical baseline for time-series forecasting.

**LSTM**

A recurrent neural network model trained on sequential input windows.

Architecture:

```
Input → LSTM → Dense → 24-step forecast
```

Training uses:

- Mean Squared Error loss
- Adam optimizer
- Early stopping based on validation loss

### Evaluation

Models are evaluated using a **chronological train / validation / test split**.

Metrics are computed on the **original scale of the target variable** using inverse scaling.

Evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

Performance is reported:

- for the entire forecast horizon
- for each individual forecast step

This allows comparison of models across short- and long-term forecasts.

## Results

The final comparison focuses on three representative models:

- **Naive baseline**
- **Ridge regression**
- **LSTM**

A **seasonal naive baseline** was also evaluated during experimentation.  
However, its performance was consistently worse than the standard naive baseline and therefore it is not included in the final comparison plots.

### Test set performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Naive | 50.67 | 85.37 | 0.32 |
| Ridge | **46.16** | **71.88** | **0.515** |
| LSTM | 46.79 | 73.77 | 0.489 |

### Horizon-wise performance

#### MAE per forecast horizon

![MAE per horizon](reports/figures/mae_per_horizon.png)

#### R² per forecast horizon

![R² per horizon](reports/figures/r2_per_horizon.png)

### Discussion

Several observations stand out from the results:

- **Ridge regression achieved the best overall performance**, particularly for short-term forecasts.
- **The LSTM model remained competitive**, but did not outperform the linear baseline on the full test horizon.
- The **performance gap between Ridge and LSTM is relatively small**, suggesting that the dataset's temporal dynamics can largely be captured by a regularized linear model.
- **Forecasting error increases with the prediction horizon** for all models, which is typical in multi-step forecasting tasks.
- While Ridge performs better for short horizons, the **LSTM becomes increasingly competitive for longer forecast horizons**, where capturing temporal dependencies may be more beneficial.

Overall, the results highlight the importance of **strong classical baselines in time-series forecasting**, as simpler models can remain highly competitive even when compared to neural networks.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/przemekwarnel/air-quality-forecasting.git
cd air-quality-forecasting
pip install -r requirements.txt
```

## How to Run

Run baseline models:

```bash
PYTHONPATH=src python -m air_quality_forecasting.run_baselines
```

Train the Ridge regression model:

```bash
PYTHONPATH=src python -m air_quality_forecasting.run_linear_model
```

Train the LSTM model:

```bash
PYTHONPATH=src python -m air_quality_forecasting.run_rnn \
  --config configs/experiments/lstm_baseline.yaml
``` 

Generate comparison plots:
```bash
PYTHONPATH=src python -m air_quality_forecasting.generate_plots \
  --lstm-experiment lstm_baseline
```

## Reproducing Experiments

All experiments are controlled via configuration files located in:

`configs`

Example configurations include:

- `configs/experiments/lstm_baseline.yaml`
- `configs/experiments/lstm_dropout.yaml`
- `configs/experiments/lstm_small_lr.yaml`

Each configuration specifies:

- model hyperparameters
- training settings
- dataset parameters

Experiment results are automatically saved to:

`reports/experiments`

as JSON files containing evaluation metrics. 

## Future Work

Possible extensions of this project include:

- testing additional neural architectures (e.g., GRU or Transformer-based models)
- incorporating additional meteorological features
- exploring probabilistic forecasting methods
- performing systematic hyperparameter optimization
- evaluating models on additional air quality datasets
