import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

employee_df = pd.read_csv("employee_turnover.csv")

x = employee_df.drop(columns=["Employee_Turnover"])
y = employee_df["Employee_Turnover"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

l1_model = LogisticRegression(penalty="l1", max_iter=1000, solver="liblinear")
l1_model.fit(x_train, y_train)
y_pred = l1_model.predict(x_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred)}, Precision : {precision_score(y_test, y_pred)}")

l2_model = LogisticRegression(penalty="l2", max_iter=1000)
l2_model.fit(x_train, y_train)
y_pred = l2_model.predict(x_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred)}, Precision : {precision_score(y_test, y_pred)}")