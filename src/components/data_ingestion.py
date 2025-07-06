import os
import sys
from src.exception import CustomException  # Import custom exception for error handling
from src.logger import logging            # Import logging for tracking events
import pandas as pd                       # Import pandas for data manipulation
from sklearn.model_selection import train_test_split  # Import for splitting data
from dataclasses import dataclass          # Import dataclass for configuration class
from src.components.data_transformation import DataTransformationConfig  # Import data transformation config
from src.components.data_transformation import DataTransformation  # Import data transformation class
from src.components.model_trainer import ModelTrainerConfig   # Import model trainer config
from src.components.model_trainer import ModelTrainer  # Import model trainer class

@dataclass
class DataIngestionConfig:
    # Path to save the training data CSV
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    # Path to save the testing data CSV
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    # Path to save the raw data CSV
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        # Initialize configuration for data ingestion.
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")  # Log start of data ingestion
        logging.info(f"Data Ingestion Config: {self.ingestion_config}")  # Log config details
        try:
            # Read the dataset from the specified CSV file
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Dataset read as pandas dataframe")  # Log successful read

            # Ensure the directory for saving artifacts exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw dataset to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")  # Log raw data save

            # Split the dataset into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training set to a CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Save the testing set to a CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train and Test data saved")  # Log train/test save
            logging.info("Ingestion of the Data is completed")  # Log completion
            # Return the paths to the train and test data files
            return (
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                )

        except Exception as e:
            # Raise a custom exception with error details if any step fails
            raise CustomException(e, sys)
#'''
if __name__ == "__main__":
    # If this script is run directly, create a DataIngestion object
     # Start the data ingestion process
    try:
        obj = DataIngestion()
    
        train_data,test_data = obj.initiate_data_ingestion()
        # Print a success message
        data_transformation = DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data, test_data)
        print("Data Ingestion completed successfully")
        modeltrainer= ModelTrainer()
        print("modeltrainer completed successfully")
        #ModelTrainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr)
        print("ModelTrainer.initiate_model_trainer completed successfully")
        print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
    except Exception as e:
        logging.info("Error in Data Ingestion")
        raise CustomException(e, sys)
    

#'''
# End of code
# End of file
