# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Preprocess Data: Import dataset, drop irrelevant columns, and encode categorical features using LabelEncoder.

2. Select Features and Target: Separate independent variables (x) and the target (y).

3. Split Data: Divide data into training and testing sets (80% training, 20% testing).

4. Train Model: Fit a Logistic Regression model on the training data.

5. Evaluate and Predict: Assess model performance and use it to predict new student placement status.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SUDARSHANAN U  
RegisterNumber: 212224230276
import pandas as pd 
data=pd.read_csv("/content/drive/MyDrive/Placement_Data.csv") 
data.head()
data1=data.copy() 
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum()
from sklearn .preprocessing import LabelEncoder 
le=LabelEncoder() 
data1["gender"]=le.fit_transform(data1["gender"]) 
data1["ssc_b"]=le.fit_transform(data1["ssc_b"]) 
data1["hsc_b"]=le.fit_transform(data1["hsc_b"]) 
data1["hsc_s"]=le.fit_transform(data1["hsc_s"]) 
data1["degree_t"]=le.fit_transform(data1["degree_t"]) 
data1["workex"]=le.fit_transform(data1["workex"]) 
data1["specialisation"]=le.fit_transform(data1["specialisation"]) 
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1] 
x 
y=data1["status"] 
y 
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test) 
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
print(classification_report1) 
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
![Screenshot 2025-03-25 035619](https://github.com/user-attachments/assets/cc16314f-3d02-43d4-8dc6-beb20b800362)
![Screenshot 2025-03-25 035613](https://github.com/user-attachments/assets/d3da4e76-06d4-49b0-a80b-52e1ebc3d187)
![Screenshot 2025-03-25 035700](https://github.com/user-attachments/assets/959ff6ed-6817-4700-b647-a3dd0445f94c)
![Screenshot 2025-03-25 035655](https://github.com/user-attachments/assets/039426a1-c38e-4cd8-955d-bc09dcb8754d)
![Screenshot 2025-03-25 035651](https://github.com/user-attachments/assets/c365330a-8a15-410a-9f02-3e27c939959b)
![Screenshot 2025-03-25 035646](https://github.com/user-attachments/assets/1cfa98d8-5d98-4978-a12c-14261d89cb72)
![Screenshot 2025-03-25 035641](https://github.com/user-attachments/assets/9f679997-3dcd-4ad4-a015-129e359c0b76)
![Screenshot 2025-03-25 035634](https://github.com/user-attachments/assets/752fd087-20bb-45cf-bd8e-43fb52ca930e)
![Screenshot 2025-03-25 035624](https://github.com/user-attachments/assets/46891037-160c-43b1-83d0-b554bdbd92b7)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
