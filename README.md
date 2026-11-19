# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, drop unnecessary columns, and encode categorical variables.
2.Define the features (X) and target variable (y).
3.Split the data into training and testing sets.
4.Train the logistic regression model, make predictions, and evaluate using accuracy and other. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Hema Ashvitha K
RegisterNumber:  212224020015
*/
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)

## Output:
<img width="1610" height="558" alt="image" src="https://github.com/user-attachments/assets/085f6563-229b-4b50-b598-ce184345240a" />
<img width="1458" height="528" alt="image" src="https://github.com/user-attachments/assets/4d4cceb8-a2cc-454a-abe0-41edad55a3ce" />
FIT TRANSFORM
<img width="1448" height="1007" alt="image" src="https://github.com/user-attachments/assets/44fe982b-54de-4b0e-b940-57b4f5237809" />
LOGISTIC REGRESSION
<img width="1595" height="563" alt="image" src="https://github.com/user-attachments/assets/0bf4b58f-2223-4e08-bd9f-a9c9316f3c77" />
ACCURACY SCORE
<img width="1585" height="312" alt="image" src="https://github.com/user-attachments/assets/53df4017-9227-4766-a420-3cfd7f1c6dff" />
CONFUSION MATRIX
<img width="1617" height="387" alt="image" src="https://github.com/user-attachments/assets/16043db3-0c0d-4911-926b-cc5778886c1c" />
CLASSIFICATION REPORT & PREDICTION
<img width="1615" height="758" alt="image" src="https://github.com/user-attachments/assets/1c166724-065b-4cf5-af5b-321bfaaded78" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
