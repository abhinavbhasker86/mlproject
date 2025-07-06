## File: src/components/data_transformation.py
# This file is part of a machine learning project that includes data transformation components. 
# It defines a class for transforming data, including scaling and encoding features, and saving the transformation pipeline.

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_cols = ["writing_score", "reading_score"]
            categorial_cols = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
                ]

            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehotencoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))  # with_mean=False for sparse matrix support
            ])  

            logging.info(f"Numerical columns: {numerical_cols}")
            logging.info(f"Categorical columns: {categorial_cols}") 
            logging.info("Numerical columns scaling and categorical columns encoding completed")    
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorial_cols)
                ]
            )   
            logging.info("Preprocessor object created successfully")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try: 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)   

            logging.info("Train and test data read successfully")
            logging.info(f"Train Dataframe Shape: {train_df.shape}")
            logging.info(f"Test Dataframe Shape: {test_df.shape}")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Preprocessing object obtained successfully")
            target_column_name = "math_score"
            numerical_cols = ["writing_score", "reading_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            logging.info("Splitting input and target features from training dataframe")
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]    
            
            logging.info("Applying preprocessing object on training and testing dataframes")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("np.c_ conversion of target feature completed")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] ##Abhi
            
            logging.info("Preprocessing completed successfully")
            

            # Save the preprocessor object to a file
            logging.info("Saving preprocessing object")
            save_object  (
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                # "target_encoder": target_encoder
            ) 
            
            #os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
            

        except Exception as e:
            raise CustomException(e, sys)
          
