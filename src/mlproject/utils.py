import os
import sys

sys.path.append(os.path.abspath("src"))
from mlproject.exception import CustomException
from mlproject.logger import logging
import pandas as pd
import pickle
import numpy as np


def read_csv_data():
    logging.info("Reading CSV dataset started")
    try:
        df =  pd.read_csv('bigmart.csv')
        logging.info("Reading data completed")
        print(df.head())
        return df
    except Exception as ex:
        raise CustomException("ex")

def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok= True)
        
        with open(file_path , "wb") as file_obj:
            pickle.dump(obj , file_obj)
            
    except Exception as e:
        raise CustomException(e , sys)
    