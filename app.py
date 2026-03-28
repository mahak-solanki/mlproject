# from src.mlproject.logger import logging

# if __name__ == "__main__":
#     logging.info("The execution has started")


import sys
import os

sys.path.append(os.path.abspath("src"))

from mlproject.logger import logging
from mlproject.exception import CustomException
from mlproject.components.data_ingestion import DataIngestion
from mlproject.components.data_ingestion import DataIngestionConfig
from mlproject.components.data_transformation import DataTransformation , DataTransformationConfig
from mlproject.components.model_trainer import ModelTrainer , ModelTrainerConfig



if __name__== "__main__":
    logging.info("the execution is started")
    
    try:
        data_ingestion = DataIngestion()
        train_data_path , test_data_path = data_ingestion.initiate_data_ingestion()
        
        data_transformation = DataTransformation()
        train_arr , test_arr,_ = data_transformation.initiate_data_transformation(train_data_path , test_data_path )
        
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr , test_arr ))
        
        
    except Exception as e:
        logging.info("Custome Exception")
        raise CustomException(e , sys)
    
    