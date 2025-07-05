'''
# File: src/components/data_ingestion.py
# This code is part of a machine learning project that handles data ingestion, which is the process of reading data from a source, 
# processing it, and saving it in a format suitable for training machine learning models. The code is structured to be modular and 
# reusable, with clear logging and error handling.
# The `DataIngestion` class is responsible for reading a dataset from a CSV file, splitting it into training and testing sets, 
# and saving these sets to specified paths. It uses the `pandas` library for data manipulation and `sklearn` for splitting the dataset. 
# The configuration for file paths is managed using a dataclass, which makes it easy to modify paths if needed.
# The code also includes logging statements to track the progress of the data ingestion process, which is crucial for debugging 
# and monitoring in production environments. It raises a custom exception if any errors occur during the ingestion process, providing 
# detailed information about the error and the context in which it occurred.
# The script can be run directly to execute the data ingestion process, and it prints a success message upon completion. 
# This makes it easy to integrate into a larger machine learning pipeline, where data ingestion is often the first step before model training and evaluation.
# The code is designed to be run in a Python environment with the necessary libraries installed, as specified in the `requirements.txt` file. It assumes that the dataset is available at the specified path and that the necessary directories for saving artifacts exist or can be created.
# The code is also structured to be easily testable, with the main functionality encapsulated in the `DataIngestion` class and the `initiate_data_ingestion` method. This allows for unit testing and integration testing to ensure that the data ingestion process works as expected.
# This code is a complete implementation of a data ingestion component in a machine learning project, providing a solid foundation for further development and integration with other components such as data preprocessing, model training, and evaluation.
# File: src/components/data_ingestion.py  
''' 
import os
import sys
from src.exception import CustomException  # Import custom exception for error handling
from src.logger import logging            # Import logging for tracking events
import pandas as pd                       # Import pandas for data manipulation
from sklearn.model_selection import train_test_split  # Import for splitting data
from dataclasses import dataclass          # Import dataclass for configuration class

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
        # Initialize configuration for data ingestion
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")  # Log start of data ingestion
        logging.info(f"Data Ingestion Config: {self.ingestion_config}")  # Log config details
        try:
            # Read the dataset from the specified CSV file
            df = pd.read_csv('notebook/data/stud.csv')
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
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            # Raise a custom exception with error details if any step fails
            raise CustomException(e, sys) from e    

if __name__ == "__main__":
    # If this script is run directly, create a DataIngestion object
    obj = DataIngestion()
    # Start the data ingestion process
    obj.initiate_data_ingestion()
    # Print a success message
    print("Data Ingestion completed successfully")
# End of code
# End of file