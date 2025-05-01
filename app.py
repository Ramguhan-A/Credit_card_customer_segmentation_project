from flask import Flask,request, render_template
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html",prediction=None)

@app.route('/predict',methods=['GET','POST'])
def predict():
    try:
         data = CustomData(
             
            BALANCE=float(request.form['BALANCE']),
            BALANCE_FREQUENCY=float(request.form['BALANCE_FREQUENCY']),
            PURCHASES=float(request.form['PURCHASES']),
            ONEOFF_PURCHASES=float(request.form['ONEOFF_PURCHASES']),
            INSTALLMENTS_PURCHASES=float(request.form['INSTALLMENTS_PURCHASES']),
            CASH_ADVANCE=float(request.form['CASH_ADVANCE']),
            PURCHASES_FREQUENCY=float(request.form['PURCHASES_FREQUENCY']),
            ONEOFF_PURCHASES_FREQUENCY=float(request.form['ONEOFF_PURCHASES_FREQUENCY']),
            PURCHASES_INSTALLMENTS_FREQUENCY=float(request.form['PURCHASES_INSTALLMENTS_FREQUENCY']),
            CASH_ADVANCE_FREQUENCY=float(request.form['CASH_ADVANCE_FREQUENCY']),
            CASH_ADVANCE_TRX=float(request.form['CASH_ADVANCE_TRX']),
            PURCHASES_TRX=float(request.form['PURCHASES_TRX']),
            CREDIT_LIMIT=float(request.form['CREDIT_LIMIT']),
            PAYMENTS=float(request.form['PAYMENTS']),
            MINIMUM_PAYMENTS=float(request.form['MINIMUM_PAYMENTS']),
            PRC_FULL_PAYMENT=float(request.form['PRC_FULL_PAYMENT']),
            TENURE=float(request.form['TENURE'])
         )
         
         final_data = data.get_data_as_dataframe()
         pipeline = PredictPipeline()
         pred = pipeline.predict(final_data)
         
         cluster_label_map = {
            0: "Highly Active User",
            1: "Inactive User",
            2: "High Spender",
            3: "Cash Advance Takers",
        }
         
         cluster_number = int(pred[0])
         readable_label = cluster_label_map.get(cluster_number)
         
         return render_template("index.html", prediction=readable_label)
         
        #  return render_template("index.html",prediction=pred[0])
        
    except Exception as e:
        return render_template("index.html", prediction=f"Error occurred: {e}")
     
     
 
if __name__ == "__main__":
     app.run(debug=True)