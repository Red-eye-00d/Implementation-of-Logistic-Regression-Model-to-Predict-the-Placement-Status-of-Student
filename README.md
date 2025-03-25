# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data progression
2. Model Training
3. Prediction
4. Evaluation

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SUDHARSANAN U
RegisterNumber:  212224230276

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('/content/drive/MyDrive/Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(['sl_no','salary'],axis=1)
data1.head()

data1.isnull()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
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

![Screenshot 2025-03-25 093451](https://github.com/user-attachments/assets/577b3099-cd2d-4e63-b4ea-94dd78b98885)

![Screenshot 2025-03-25 093508](https://github.com/user-attachments/assets/6515b37e-9a9f-4ed8-b2fb-1bbf7a6fba9d)

![Screenshot 2025-03-25 093518](https://github.com/user-attachments/assets/89ca9ac5-1e3c-4f6e-8aff-65893e05f558)

![Screenshot 2025-03-25 093525](https://github.com/user-attachments/assets/fa130c5c-2485-49f7-afa6-757e8a6f4f95)

![Screenshot 2025-03-25 093537](https://github.com/user-attachments/assets/9cb45497-5a88-4b1a-a01e-61ef7dc3ad4e)

![Screenshot 2025-03-25 093544](https://github.com/user-attachments/assets/0f898137-3803-4e87-bb0e-ba1d9743d59d)

![Screenshot 2025-03-25 093550](https://github.com/user-attachments/assets/632ce3c1-d5c6-4ab3-9470-d61fed182a5b)

![Screenshot 2025-03-25 093555](https://github.com/user-attachments/assets/d8ef09d6-6bde-445e-89df-5a906b55f61c)

![Screenshot 2025-03-25 093602](https://github.com/user-attachments/assets/bcaf7a13-9d78-4869-b75e-ad391331dcf1)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
