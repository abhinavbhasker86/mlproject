import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
import pickle
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
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    """
    Evaluate multiple regression models and return their R2 scores.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file using pickle.
    """
    try:
        logging.info(f"Loading object from {file_path}")
        logging.info(f"----------Going to Loading object----------")
        # Open the file in binary read mode and load the object
        with open(file_path, "rb") as file_obj:
            logging.info(f"Object loaded successfully from {file_path}")
            return pickle.load(file_obj)
        
        
        #return obj
    
    except Exception as e:
        raise CustomException(e, sys)
    