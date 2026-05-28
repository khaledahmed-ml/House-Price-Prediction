import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load data
df=pd.read_csv("House_price.csv")
# Define features and target
X=df.drop(columns=["Price","Address"])
Y=df["Price"]
# Split and scale data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3,random_state=42)
lr=LinearRegression()
scale=StandardScaler()
X_train_scaled=scale.fit_transform(X_train)
X_test_scaled=scale.transform(X_test)
# Train model
lr.fit(X_train_scaled,Y_train)
Y_pred=lr.predict(X_test_scaled)
#--------------------------
# Measure m & b
print(f"m= {lr.coef_}") # how much the price changes when this feature increases by 1
print(f"b= {lr.intercept_}") # starting value
print("#"*50)
#--------------------------
# Performance of Train & Test (R^2)
print(f"The Performance of train: {lr.score(X_train_scaled,Y_train)}")
print(f"The Performance of test: {lr.score(X_test_scaled,Y_test)}")
print("#"*50)
#--------------------------
#Ranges:
ranges = {
"income": (X["Avg. Area Income"].min(), X["Avg. Area Income"].max()),
"age": (X["House Age"].min(), X["House Age"].max()),
"rooms": (X["Number of Rooms"].min(), X["Number of Rooms"].max()),
"bedrooms": (X["Number of Bedrooms"].min(), X["Number of Bedrooms"].max()),
"pop": (X["Area Population"].min(), X["Area Population"].max()),
}
#--------------------------
# Inputs 
# Why we using float? Using 'int' may cause errors or lose precision, float is safer for ML inputs
def get_input(name, min_val, max_val):
    while True:
        val = float(input(f"{name}: "))
        if min_val <= val <= max_val:
            return val
        else:
            print(f"Enter {name} between {min_val:.2f} and {max_val:.2f}")
# Interactive prediction
income = get_input("Income", *ranges["income"])
age = get_input("Age", *ranges["age"])
rooms = get_input("Rooms", *ranges["rooms"])
bedrooms = get_input("Bedrooms", *ranges["bedrooms"])
population = get_input("Population", *ranges["pop"])
#--------------------------
# Predict
predict=scale.transform(np.array([[income,age,rooms,bedrooms,population]]))
print(f"The prediction of House Price: {lr.predict(predict)[0]:,.4f} $")
