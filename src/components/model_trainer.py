
from src.entity.config_entity import ModelTrainerConfig
from src.logger.custom_logging import logger
from sklearn.model_selection import train_test_split
import sys,joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from src.utils.utlis import *
from src.exceptions.expection import CustomException


class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config = config

    @staticmethod
    def save_model(path: Path, model: BaseEstimator):
        """Save the trained model to the specified path."""
        try:
            joblib.dump(model, path)
            print(f"Model saved at {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            logger.error(f"Error saving model: {e}")
            raise CustomException(e, sys)  

    def initate_model_trainer(self):
        train_data=pd.read_csv(self.config.training_data_path)
        test_data=pd.read_csv(self.config.testing_data_path)
        try:
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values   


            models = {
                "Random Forest": RandomForestRegressor(),
                "SVR": SVR()
            }
            

            # Evaluate models
            model_report = final_model(models, X_train, X_test, y_train, y_test)
            print(model_report)
            print('\n====================================================================================\n')
            logger.info(f'Model Report: {model_report}')

            # Get the best model score
            best_model_score = max(model_report.values())
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name is: {best_model_name}, Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logger.info(f"Best model found, Model Name is {best_model_name}, Accuracy Score: {best_model_score}")

            # Save the best model
            self.save_model(path=self.config.train_model_path,model=best_model)
        except Exception as e:
            logger.error(f'Error occurred: {e}')
            raise CustomException(e, sys)