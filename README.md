# Number-Recognition1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data with features (socio-economic status, age, gender) and a binary label (0 or 1 for safe or not safe).
data = {
    'SocioEconomicStatus': [1, 2, 3, 2, 1, 3, 2, 3, 1, 2],
    'Age': [25, 30, 35, 28, 22, 40, 32, 38, 26, 29],
    'Gender': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'Safe': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
}

df = pd.DataFrame(data)

# Split the data into features and labels.
X = df[['SocioEconomicStatus', 'Age', 'Gender']]
y = df['Safe']

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a logistic regression model.
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set.
y_pred = model.predict(X_test)

# Calculate the accuracy of the model.
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

