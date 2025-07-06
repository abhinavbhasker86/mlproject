import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Model Trainer started")
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor(),
                'KNN': KNeighborsRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False),
                'AdaBoost': AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models)
            '''
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2_square = r2_score(y_test, y_pred)
                model_report[model_name] = r2_square
                logging.info(f"{model_name} R2 Score: {r2_square}")
            '''
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            #best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    f"{best_model_name} Model performance is not satisfactory: {best_model_score}",
                    sys
                )

            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")

            #preprocessing_obj = load_object(file_path=preprocessor_path)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            #predicted_data = best_model.predict(X_test)
            #r2_square = r2_score(y_test, predicted_data)
            return best_model_name, best_model_score #, r2_square

        except Exception as e:
            raise CustomException(e, sys) from e