import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

heart_df = pd.read_csv("heart.csv")

x = heart_df.drop(columns=["target"])
y = heart_df["target"]

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
print(f"Precision Score: {precision_score(y_test, y_pred)}")