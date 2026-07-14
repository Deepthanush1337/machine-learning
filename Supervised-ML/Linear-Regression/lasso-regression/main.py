import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

insurance_data = pd.read_csv("insurance.csv")

x = insurance_data.drop(columns=["charges"])
y = insurance_data["charges"]

x["smoker"] = x["smoker"].map({"yes":1,"no":0 })
x["sex"] = x["sex"].map({"male":1, "female":0})

x = pd.get_dummies(x, columns=["region"], drop_first=True, dtype=int)
x["age_smoker"] = x["age"] * x["smoker"]
x["bmi_smoker"] = x["bmi"] * x["smoker"]
# Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Train model
alphas = [0.001, 0.1, 1, 2, 5, 10, 20, 50, 100]
mses = []
import seaborn as sns
import matplotlib.pyplot as plt

for a in alphas:
    model = Lasso(alpha=a)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE for {a}: {mse}")
    mses.append(mse)

# Cross validation
from sklearn.linear_model import LassoCV
alphas = [0.001, 0.1, 1, 2, 5, 10, 20, 50, 100]
LassoCV_model = LassoCV(
    alphas=alphas,
    cv=5, 
    max_iter=1000, 
    random_state=42
)

LassoCV_model.fit(x_train, y_train)
print("best alpha: ", LassoCV_model.alpha_)
y_pred = LassoCV_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"mse: {mse}")
sns.lineplot(x=alphas, y=mses, marker="o")
plt.show()