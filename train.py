
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# 1. Load the California Housing dataset
data = fetch_california_housing()

X = data.data
y = data.target

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")


# 2. Split into 80% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


# 3. Configure the MLflow experiment
mlflow.set_experiment("California Housing Experiments")

# Each tuple contains:
# Run name, max_depth, learning_rate
experiments = [
    ("Run 1", 3, 0.1),
    ("Run 2", 5, 0.05),
    ("Run 3", 7, 0.01),
]

results = []


# 4. Train and track each experiment
for run_name, max_depth, learning_rate in experiments:

    with mlflow.start_run(run_name=run_name):

        # Create the model
        model = GradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)

        # Predict on validation data
        predictions = model.predict(X_val)

        # Calculate validation metrics
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)

        # Log parameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)

        # Log metrics
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Log the trained model as an artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        # Save results for comparison
        results.append({
            "Run": run_name,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        print(f"\n{run_name} completed")
        print(f"max_depth: {max_depth}")
        print(f"learning_rate: {learning_rate}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")


# 5. Identify the best run using RMSE
best_run = min(results, key=lambda result: result["RMSE"])

print("\n" + "=" * 45)
print("EXPERIMENT COMPARISON")
print("=" * 45)

for result in results:
    print(
        f"{result['Run']}: "
        f"Depth={result['max_depth']}, "
        f"LR={result['learning_rate']}, "
        f"RMSE={result['RMSE']:.4f}, "
        f"MAE={result['MAE']:.4f}, "
        f"R2={result['R2']:.4f}"
    )

print("\nBEST MODEL")
print(f"Run: {best_run['Run']}")
print(f"RMSE: {best_run['RMSE']:.4f}")
print(f"Parameters: depth={best_run['max_depth']}, "
      f"learning_rate={best_run['learning_rate']}")