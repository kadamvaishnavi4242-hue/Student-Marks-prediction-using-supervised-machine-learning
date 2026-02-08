import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix,mean_absolute_error,mean_squared_error,root_mean_squared_error
df=pd.read_csv(r"C:\Users\kadam\Downloads\Student Data.CSV",encoding="latin1")
X=df[["Hours"]]
y=df[["Score"]]
model=LinearRegression()
model.fit(X,y)
predicted=model.predict(X)
mae=mean_absolute_error(y,predicted)
mse=mean_squared_error(y,predicted)
rmse=np.sqrt(mse)
print("mean absolute error is",mae)
print("mean squared error is",mse)
print("root mean squared error is",rmse)
predicted_hour=float(input("enter the hour"))
new_pred=model.predict([[predicted_hour]])
print(f"based on your prediction you may score around{new_pred}")