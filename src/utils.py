import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill

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