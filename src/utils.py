import os
import sys

from src.exception import CustomException
from src.logger import logging

import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
         raise CustomException(e,sys)
     
def evaluate_model(X_train,X_test,y_train, y_test ,models, params):
    
    try:
        report = {}
        for model_name,model in models.items():
            para = params.get(model_name,{})
            
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train,y_train)
            
            best_model = gs.best_estimator_
            
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            train_model_score = accuracy_score(y_train,y_train_pred)
            test_model_score = accuracy_score(y_test,y_test_pred)
            
            report[model_name] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
