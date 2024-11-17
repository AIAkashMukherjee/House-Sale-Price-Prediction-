from src.config.configuration import EvaluationConfig
from src.utils.utlis import *
from src.constants import *
import mlflow
import joblib
import pandas as pd

class ModelEvaluation:
    def __init__(self,config:EvaluationConfig):
        self.config=config
        self.scores={}

    def save_score(self):
        save_json(path=Path("scores.json"), data=self.scores) 

    def evaluate_model(self):
            # train_data=pd.read_csv(self.config.training_data_path)
            test_data=pd.read_csv(self.config.test_data)
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values   

            model=joblib.load(self.config.model_path)
            y_pred=model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            self.scores["R2 Score"] = r2
            return model

    def log_with_mlflow(self,model):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        with mlflow.start_run():
            mlflow.log_param("Model Path", self.config.model_path)
            mlflow.log_metric("R2 Score", self.scores["R2 Score"])

            mlflow.sklearn.log_model(model, "random_forest_model")

    def evaluation(self):
        model = self.evaluate_model()

        # Save scores locally
        self.save_score()

        # Log evaluation with MLflow
        self.log_with_mlflow(model)
            