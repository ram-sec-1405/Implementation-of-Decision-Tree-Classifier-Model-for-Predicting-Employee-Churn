# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step-1: 
Import the required libraries.
### Step-2:
2. Load the dataset.
### Step-3:
3. Define X and Y array.
### Step-4:
4. Define a function for costFunction,cost and gradient.
### Step-5:
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value. 

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: RAMPRASATH R
RegisterNumber: 212223220086

import pandas as pd
data=pd.read_csv("Employee.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
print(x.head())

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
![image](https://github.com/SridharShyam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871368/f0e59a24-d642-4731-960f-399888d45c97)

![image](https://github.com/SridharShyam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871368/ef575661-e899-4530-8322-47295ef6097b)

![image](https://github.com/SridharShyam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871368/0788ddd9-f5c3-4318-a56e-77d57a2d48b2)

![image](https://github.com/SridharShyam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871368/47aae5eb-678c-4e0b-b0f3-40d6a182f3f4)
#### Accuracy:
![image](https://github.com/SridharShyam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871368/04ea367c-4b42-443c-ab1e-e02fab150733)
#### Prediction:
![image](https://github.com/SridharShyam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144871368/edfb04a5-b4e9-47a9-87c4-37d423c39129)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
