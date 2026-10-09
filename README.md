# MLflow Experiment Tracking — California Housing

## Project Overview

This project demonstrates experiment tracking using MLflow for a regression task on the California Housing dataset. Three models were trained with different hyperparameter configurations, and their parameters, evaluation metrics, and trained model artifacts were logged using MLflow.

## Objectives

* Train three models with different hyperparameters.
* Log parameters and evaluation metrics using MLflow.
* Save trained models as MLflow artifacts.
* Compare model performance and select the best run based on the lowest RMSE.

## Dataset

* **Dataset:** California Housing
* **Source:** Scikit-learn built-in dataset
* **Features:** 8 input features
* **Total records:** 20,640
* **Training set:** 80% (16,512 records)
* **Validation set:** 20% (4,128 records)
* **Random state:** 42

## Technologies Used

* Python
* Scikit-learn
* MLflow
* NumPy
* Pandas

## Experiment Configurations

| Run   | max_depth | learning_rate |
| ----- | --------: | ------------: |
| Run 1 |         3 |           0.1 |
| Run 2 |         5 |          0.05 |
| Run 3 |         7 |          0.01 |

## Results

| Run   | max_depth | learning_rate |   RMSE |    MAE |     R² |
| ----- | --------: | ------------: | -----: | -----: | -----: |
| Run 1 |         3 |           0.1 | 0.5422 | 0.3716 | 0.7756 |
| Run 2 |         5 |          0.05 | 0.5198 | 0.3531 | 0.7938 |
| Run 3 |         7 |          0.01 | 0.6973 | 0.5311 | 0.6290 |

## Best Model

**Run 2** achieved the lowest RMSE of **0.5198**, making it the best-performing model among the three tested configurations.

Its parameters were:

* `max_depth = 5`
* `learning_rate = 0.05`

Run 2 also achieved the lowest MAE and highest R² among the tested runs.

## MLflow Tracking

MLflow was used to record the following for each run:

* **Parameters:** `max_depth`, `learning_rate`
* **Metrics:** RMSE, MAE, R²
* **Artifacts:** Saved trained model (`model.skops`)

## How to Run

1. Clone or download this repository.

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:

   ```bash
   python train.py
   ```

4. Start the MLflow UI in a separate terminal:

   ```bash
   mlflow ui --host 127.0.0.1 --port 5000
   ```

5. Open `http://127.0.0.1:5000` in your browser and select the **California Housing Experiments** experiment.

## Conclusion

The experiment demonstrates how MLflow can track hyperparameters, evaluate model performance, and save trained model artifacts. Among the three configurations, Run 2 performed best based on the lowest RMSE.
