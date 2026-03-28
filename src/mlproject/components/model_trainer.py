import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

sys.path.append(os.path.abspath("src"))
from mlproject.exception import CustomException
from mlproject.logger import logging
from mlproject.utils import evaluate_model , save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts" , "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array , test_array):
        try:
            logging.info("Split training and test input data")
            x_train , y_train , x_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:, -1]
            )
            #models for training
            models = {
                "Linear Regressor" : LinearRegression(),
                "Decision Tree" : DecisionTreeRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "KNearestNeighbor" : KNeighborsRegressor()
            }
            # hyperparameter tuning
            params = {
                "Decision Tree" : {
                    "criterion" :["squared_error" , "friedman_mse" , "absolute_error" , "poisson"],
                    "splitter" : ["best" , "random"]
                },
                "Linear Regressor" : {},
                "XGBRegressor" : {
                    "learning_rate" :[.001 , 0.05 , 0.1] ,
                    "n_estimators" : [8, 16 , 32 , 64 , 128 , 256]
                },
                "KNearestNeighbor" :{
                    "n_neighbors" : [1 ,2 ,3, 4, 5,6, 7],
                    "algorithm" : ["auto" , "ball_tree" , "kd_tree" , "brute"]
                }
                
            }
            model_report :dict= evaluate_model(x_train ,y_train , x_test , y_test , models , params)
            
            #to get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            #to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]            
            best_model =  models[best_model_name]
            
            print("This is the best model")
            print(best_model_name)
            
            if best_model_score<0.6 :
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted =  best_model.predict(x_test)
            
            r2Score = r2_score(y_test , predicted)
            return r2Score
        
            
        except Exception as e:
            raise CustomException(e , sys)