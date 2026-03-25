import os
import sys

sys.path.append(os.path.abspath("src"))
from mlproject.exception import CustomException
from mlproject.logger import logging
import pandas as pd

def read_csv_data():
    logging.info("Reading CSV dataset started")
    try:
        df =  pd.read_csv('bigmart.csv')
        logging.info("Reading data completed")
        print(df.head())
        return df
    except Exception as ex:
        raise CustomException("ex")
