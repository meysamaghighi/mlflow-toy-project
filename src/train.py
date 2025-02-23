import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
X = df.drop(columns=["species"])
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(n_estimators=100):
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

        print(f"Model trained with {n_estimators} trees. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model(100)  # Train with 100 trees
