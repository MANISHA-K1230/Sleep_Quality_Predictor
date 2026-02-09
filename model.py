import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Sample dataset (replace with Kaggle dataset for real project)
data = {
    "sleep_duration": [5,6,7,8,4,9,6.5,7.5],
    "physical_activity": [10,30,60,90,5,100,40,70],
    "stress_level": [8,6,4,2,9,1,5,3],
    "screen_time": [120,90,60,30,150,20,80,40],
    "sleep_quality": ["Poor","Average","Good","Good","Poor","Good","Average","Good"]
}

df = pd.DataFrame(data)

X = df.drop("sleep_quality", axis=1)
y = df["sleep_quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("sleep_model.pkl", "wb"))
