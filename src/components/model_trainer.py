import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainconfig:
    Model_train_file_path = os.path.join("artifact","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.Model_train_config = ModelTrainconfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting the train and test input data... ")
            
            X_train,X_test,y_train,y_test = (
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            ) 
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier()
            }
            
            params = {
                
            }
            
            model_report: dict=evaluate_model(X_train=X_train,X_test = X_test,y_train = y_train, y_test = y_test
                                              ,models = models, params = params )
            
            best_model_name = max(model_report,key=model_report.get)
            best_model_score = model_report[best_model_name]
            
            best_model = models[best_model_name]
            
            logging.info("Finding Best model ... ")
            
            if best_model_score < 0.6:
                raise CustomException("No best model found, all scores are very less")
            
            logging.info(f"Best model found: {best_model_name} with a accuracy score: {best_model_score}")
            
            best_model.fit(X_train,y_train)
            
            save_object(
                file_path=self.Model_train_config.Model_train_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(X_test)
            
            accuracy = accuracy_score(y_test,predicted)
            return accuracy
        
        except Exception as e:
            raise CustomException(e,sys)