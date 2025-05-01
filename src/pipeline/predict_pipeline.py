import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def init(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join("artifact","model.pkl")
            preprocessor_path = os.path.join("artifact","preprocessor.pkl")
            print("Loading...")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("predicting...")
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
            
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,BALANCE:float,BALANCE_FREQUENCY:float,PURCHASES:float,ONEOFF_PURCHASES:float,INSTALLMENTS_PURCHASES:float,
                 CASH_ADVANCE:float,PURCHASES_FREQUENCY:float,ONEOFF_PURCHASES_FREQUENCY:float,PURCHASES_INSTALLMENTS_FREQUENCY:float,
                 CASH_ADVANCE_FREQUENCY:float,CASH_ADVANCE_TRX:float,PURCHASES_TRX:float,CREDIT_LIMIT:float,PAYMENTS:float,
                 MINIMUM_PAYMENTS:float,PRC_FULL_PAYMENT:float,TENURE:float):
            self.BALANCE = BALANCE
            self.BALANCE_FREQUENCY = BALANCE_FREQUENCY
            self.PURCHASES = PURCHASES
            self.ONEOFF_PURCHASES = ONEOFF_PURCHASES
            self.INSTALLMENTS_PURCHASES = INSTALLMENTS_PURCHASES
            self.CASH_ADVANCE = CASH_ADVANCE
            self.PURCHASES_FREQUENCY = PURCHASES_FREQUENCY
            self.ONEOFF_PURCHASES_FREQUENCY = ONEOFF_PURCHASES_FREQUENCY
            self.PURCHASES_INSTALLMENTS_FREQUENCY = PURCHASES_INSTALLMENTS_FREQUENCY
            self.CASH_ADVANCE_FREQUENCY = CASH_ADVANCE_FREQUENCY
            self.CASH_ADVANCE_TRX = CASH_ADVANCE_TRX
            self.PURCHASES_TRX = PURCHASES_TRX
            self.CREDIT_LIMIT = CREDIT_LIMIT
            self.PAYMENTS = PAYMENTS
            self.MINIMUM_PAYMENTS = MINIMUM_PAYMENTS
            self.PRC_FULL_PAYMENT = PRC_FULL_PAYMENT
            self.TENURE = TENURE
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "BALANCE": [self.BALANCE],
                "BALANCE_FREQUENCY": [self.BALANCE_FREQUENCY],
                "PURCHASES": [self.PURCHASES],
                "ONEOFF_PURCHASES": [self.ONEOFF_PURCHASES],
                "INSTALLMENTS_PURCHASES": [self.INSTALLMENTS_PURCHASES],
                "CASH_ADVANCE": [self.CASH_ADVANCE],
                "PURCHASES_FREQUENCY": [self.PURCHASES_FREQUENCY],
                "ONEOFF_PURCHASES_FREQUENCY": [self.ONEOFF_PURCHASES_FREQUENCY],
                "PURCHASES_INSTALLMENTS_FREQUENCY": [self.PURCHASES_INSTALLMENTS_FREQUENCY],
                "CASH_ADVANCE_FREQUENCY": [self.CASH_ADVANCE_FREQUENCY],
                "CASH_ADVANCE_TRX": [self.CASH_ADVANCE_TRX],
                "PURCHASES_TRX": [self.PURCHASES_TRX],
                "CREDIT_LIMIT": [self.CREDIT_LIMIT],
                "PAYMENTS": [self.PAYMENTS],
                "MINIMUM_PAYMENTS": [self.MINIMUM_PAYMENTS],
                "PRC_FULL_PAYMENT": [self.PRC_FULL_PAYMENT],
                "TENURE": [self.TENURE]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)