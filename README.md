# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program & Output:
```
/* Program to implement the SVM For Spam Mail Detection..
Developed by: MANIKANDAN S
RegisterNumber: 25012138
```
```
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
data=pd.read_csv("drive/MyDrive/ML/spam.csv", encoding='Windows-1252')
data
```
![image](https://github.com/user-attachments/assets/1a8adc56-080f-450e-8563-0b8b4b2610e3)
```
data.shape
```
![image](https://github.com/user-attachments/assets/cc85cb02-d338-4466-b996-52a6052f03e1)
```
x=data['v2'].values
y=data['v1'].values
x.shape
```
![image](https://github.com/user-attachments/assets/686b717a-e89f-42b5-aae4-de1097196a52)
```
y.shape
```
![image](https://github.com/user-attachments/assets/6fafd9ec-1941-4d85-aa20-85dfcbb3c173)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train
```
![image](https://github.com/user-attachments/assets/e100391d-43e1-4ebd-afae-d45924ddfa60)
```
x_train.shape
```
![image](https://github.com/user-attachments/assets/c5cf80a2-7962-4a53-bef7-f772261f88fb)
```
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/624490c3-c87f-46a2-a33c-9c31541732e9)
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
```
![image](https://github.com/user-attachments/assets/26cf8f48-eb4c-418d-a23d-6f88cafd90b4)
```
con=confusion_matrix(y_test,y_pred)
print(con)
```
![image](https://github.com/user-attachments/assets/f3f9df96-a5e6-49d5-a32d-1eca2a456236)
```
cl=classification_report(y_test,y_pred)
print(cl)
```
![image](https://github.com/user-attachments/assets/53cbbfea-8ad8-4c68-bbcd-ff5741890184)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
