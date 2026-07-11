import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

house_data = pd.read_csv("house_data.csv")
house_data = house_data.dropna()

x = house_data.drop(columns=["Id", "SalePrice"])
y = house_data["SalePrice"]

x = pd.get_dummies(
    x,
    columns=["MSZoning", "LotConfig", "BldgType", "Exterior1st"],
    drop_first=True
)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 4, test_size = 0.2)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)


n = x_test.shape[0]
p = x_test.shape[1]

adj_r2_score = 1 - (((1 - r2)*(n-1))/n-p-1)
print(x.isna().sum())
print(r2)
print(adj_r2_score)