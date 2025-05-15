# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import required libraries for data handling, preprocessing, modeling, and evaluation.
2. Load the dataset from the CSV file into a pandas DataFrame.
3. Check for null values and inspect data structure using .info() and .isnull().sum().
4. Encode the categorical "Position" column using LabelEncoder.
5. Split features (Position, Level) and target (Salary), then divide into training and test sets.
6. Train a DecisionTreeRegressor model on the training data.
7. Predict on test data, calculate mean squared error and R² score, and make a sample prediction.

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: S.Keerthika
RegisterNumber: 212223040093
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
```
```
data = pd.read_csv("Salary.csv")
```
```
data.head()
```
```
data.info()
```
```
# display the count of null values
data.isnull().sum()
```
```
# encode postion using label encoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
```
```
# defining x and y and splitting them
x = data[["Position", "Level"]]
y = data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
```
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
```
```
# predicting test values with model
y_pred = dt.predict(x_test)
```
```
mse = metrics.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error : {mse}")
```
```
r2 = metrics.r2_score(y_test, y_pred)
print(f"R Square : {r2}")
```
```
dt.predict(pd.DataFrame([[5,6]], columns=["Position", "Level"]))
```

## Output:

Head Values

![Screenshot 2025-05-14 212722](https://github.com/user-attachments/assets/3b2a14a8-b45e-4733-9b09-d244cee1d3ef)

DataFrame Info

![Screenshot 2025-05-14 212728](https://github.com/user-attachments/assets/c7e8d148-1e8e-4c2e-92f1-68431c0511c9)

Sum - Null Values

![Screenshot 2025-05-14 212733](https://github.com/user-attachments/assets/ddaeaeda-4a06-4d6c-a83e-2d023ae7f56f)

Encoding position feature

![Screenshot 2025-05-14 212740](https://github.com/user-attachments/assets/e8a229f7-5320-4948-96d9-145a4bd279e4)

Training the model

![Screenshot 2025-05-14 212746](https://github.com/user-attachments/assets/8917f5b6-f904-4545-8e73-f1c0c9db30eb)

Mean Squared Error

![Screenshot 2025-05-14 212751](https://github.com/user-attachments/assets/1753b6a7-47bc-4da2-8b60-0098b5e24419)

R2 Score

![Screenshot 2025-05-14 212756](https://github.com/user-attachments/assets/fc694185-aa72-4a3a-996a-f88f02e64102)

Final Prediction on Untrained Data

![Screenshot 2025-05-14 212801](https://github.com/user-attachments/assets/ee821d3b-db64-4680-a0ad-7056aa887a30)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
