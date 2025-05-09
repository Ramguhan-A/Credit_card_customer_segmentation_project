# Customer Segmentaiton project: 🛗

## Objective:
Segmenting Customers based on the Credit Card usage and predicting the customers behaviour

## Tools used:

$ Pandas, 
$ matplotlib, 
$ seaborn, 
$ sci_kit learn, 
$ Flask, 
$ HTML and CSS

## 📔 Notebook File 1 --> Clustering

 [Customer Segmentation Notebook](https://github.com/Ramguhan-A/Credit_card_customer_segmentation_project/blob/main/notebook/1_Customer_Segmentation.ipynb)
* Step 1: Importing Dataset (Kaggle)
* Step 2: Preprocessing
* Step 3: EDA
* Step 4: log transformation
* Step 5: StandardScaler
* Step 6: PCA using scree plot
* Step 7: KMeans Clustering using Elbow method
* Step 8: Concatinating the labels with original dataset
* Step 9: Saving the data in CSV format

## 📔 Notebook File 2 --> Predicting the behaviour

 [Understanding and Predicting Customer Behaviour Notebook](https://github.com/Ramguhan-A/Credit_card_customer_segmentation_project/blob/main/notebook/2_Understanding%20and%20Predicting%20Customer%20Behaviour.ipynb)
* Step 1: Importing the Clustered data
* Step 2: Preprocessing
* Step 3: EDA
* Step 4: Mapping the Clustered data into Human Understadable form based on EDA
* Step 5: Train Test Split
* Step 6: StandardScaler
* Step 7: Model Training and Performance Evaluation
* Step 8: Confusion matrix
  
## Result obtained:
🎯 Model	Accuracy_score
* RandomForestClassifier	0.972626
* Logistic Regression	0.963128
* Decision Tree Classifier	0.949721

### " ✅ Achieved 97.2 % Accuracy through Random Forest Classifier, Now based on the prediciton we can improve customer experience, Customer Engagement and we can provide Customized offers to our customers."
