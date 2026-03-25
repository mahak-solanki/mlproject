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

if __name__== "__main__":
    logging.info("the execution is started")
    
    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("Custome Exception")
        raise CustomException(e , sys)
    
    