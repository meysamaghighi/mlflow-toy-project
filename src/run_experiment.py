import mlflow
import optuna
from src.train import train_model

mlflow.set_experiment("mlflow_toy_experiment")

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    train_model(n_estimators)
    return mlflow.get_run(mlflow.last_active_run().info.run_id).data.metrics["accuracy"]

# Run hyperparameter tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

print(f"Best parameters: {study.best_params}")
