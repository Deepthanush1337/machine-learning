import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
insurance_data = pd.read_csv("insurance.csv")

"""sns.scatterplot(x=insurance_data["bmi"], y=insurance_data["charges"], hue=insurance_data["smoker"])
plt.show()"""

x = insurance_data.drop(columns=["charges", "region"])
y = insurance_data["charges"]

x["smoker"] = x["smoker"].map({"yes":1,"no":0 })
x["sex"] = x["sex"].map({"male":1, "female":0})

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict values
y_pred = model.predict(x_test)
print(y_pred)
print(y_test)

# Evaluate 
r_sq = r2_score(y_test, y_pred)
print(f"R Squared Error: {r_sq}")

n = x_test.shape[0]
p = x_test.shape[1]

adj_r_sq = 1 - ((1 - r_sq) * (n-1))/(n-p-1)
print(adj_r_sq)