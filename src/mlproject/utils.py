import os
import sys

sys.path.append(os.path.abspath("src"))
from mlproject.exception import CustomException
from mlproject.logger import logging
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


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
    
def evaluate_model(x_train , y_train , x_test , y_test , models , params):
    try :
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model , para , cv =3)
            gs.fit(x_train , y_train)
        
            model.set_params(**gs.best_params_)
            model.fit(x_train , y_train)
            
            y_train_pred= model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            train_model_score = r2_score(y_train , y_train_pred)
            test_model_score = r2_score(y_test , y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e , sys)
    
    