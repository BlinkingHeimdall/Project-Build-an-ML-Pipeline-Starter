# NYC Airbnb Rental Price ML Pipeline

This repository contains an MLOps pipeline for training and deploying a machine learning model to predict short-term rental prices in New York City.  

The pipeline uses:

- **Hydra** for configuration management  
- ![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue) for orchestration and experiment tracking  
- ![badge](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white) for artifact storage, model registry, and lineage tracking

The project follows the ![badge](https://img.shields.io/badge/Udacity-grey?style=for-the-badge&logo=udacity&logoColor=#5FCFEE) MLOps “Build an ML Pipeline for Short-Term Rentals” specification. The original repo can be found here:
https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter

---

## Links

- **GitHub repository:**  
  https://github.com/BlinkingHeimdall/Project-Build-an-ML-Pipeline-Starter  

- **W&B project (public):**  
  https://wandb.ai/jmadams41-western-governors-university/nyc_airbnb?nw=nwuserjmadams41

---

## Pipeline Overview

The ML pipeline is defined in `main.py` and controlled via `config.yaml`.  
It consists of the following steps:

1. **download (`get_data` component)**  
   - Downloads a sample of the data (`sample1.csv`)  
   - Logs it to W&B as `sample.csv` (`raw_data` artifact)

2. **basic_cleaning (`src/basic_cleaning`)**  
   - Filters prices to be between `etl.min_price` and `etl.max_price`  
   - Converts `last_review` to datetime  
   - Applies geolocation filtering to keep only listings within NYC bounds  
   - Outputs `clean_sample.csv` (`clean_sample` artifact)

3. **data_check (`src/data_check`)**  
   - Runs data quality tests:
     - Column names
     - Neighborhood names
     - Geolocation boundaries
     - KL divergence vs. reference dataset
     - **Custom tests:** `test_row_count`, `test_price_range`  
   - Uses `clean_sample.csv:latest` as the current sample  
   - Uses `clean_sample.csv:reference` as the reference dataset

4. **data_split (`train_val_test_split` component)**  
   - Splits the cleaned data into:
     - `trainval_data.csv`
     - `test_data.csv`  
   - Uses parameters `modeling.test_size`, `modeling.val_size`, and `modeling.stratify_by`

5. **train_random_forest (`src/train_random_forest`)**  
   - Builds a preprocessing pipeline:
     - Imputers for numeric & categorical data
     - OneHotEncoder for categoricals
     - Text features via TF-IDF (e.g., listing name)
   - Trains a Random Forest regressor using parameters from `modeling.random_forest`
   - Logs validation **MAE** and **R²** to W&B
   - Logs the trained model as `random_forest_export` (MLflow sklearn model)

6. **test_regression_model (`test_regression_model` component)**  
   - Loads the promoted model `random_forest_export:prod`
   - Evaluates it on `test_data.csv:latest`
   - Logs test metrics to W&B

---

## Configuration

All configurable parameters are in `config.yaml`.

Key sections:

~~~
etl:
  sample: "sample1.csv"
  min_price: 10
  max_price: 350

data_check:
  kl_threshold: 0.2

modeling:
  test_size: 0.2
  val_size: 0.2
  random_seed: 42
  stratify_by: "neighbourhood_group"
  max_tfidf_features: 5

  random_forest:
    n_estimators: 100
    max_depth: 15
    min_samples_split: 4
    min_samples_leaf: 3
    n_jobs: -1
    criterion: squared_error
    max_features: 0.5
    oob_score: true
~~~

![badge](https://img.shields.io/github/contributors/BlinkingHeimdall/Project-Build-an-ML-Pipeline-Starter)