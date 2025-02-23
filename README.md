# MLflow Toy Project 🚀
A simple MLflow project to log experiments and track hyperparameter tuning.

## 📌 Setup & Run
```bash
# Clone repo
git clone https://github.com/meysamaghighi/mlflow-toy-project.git
cd mlflow-toy-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py

# Run hyperparameter tuning
python src/run_experiment.py

# View MLflow UI
mlflow ui
