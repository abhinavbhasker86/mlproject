import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path,obj):
    """
    Save an object to a file using pickle.
    """
    try:
        logging.info(f"Saving object to {file_path}")

        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        # Open the file in binary write mode and dump the object
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    """
    Evaluate multiple regression models and return their R2 scores.
    """
    model_report = {}
    
    for model_name, model in models.items():
        try:
            logging.info(f"Evaluating model: {model_name}, with parameters: {param[model_name]}")
            para=param[model_name]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)			
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_r2_square = r2_score(y_train, y_train_pred)
            test_model_r2_square = r2_score(y_test, y_test_pred)
            model_report[model_name] = test_model_r2_square
            logging.info(f"Test {model_name} R2 Score: {test_model_r2_square}")
        except Exception as e:
            #logging.error(f"Error evaluating {model_name}: {e}")
            raise CustomException(e, sys)
    
    return model_report