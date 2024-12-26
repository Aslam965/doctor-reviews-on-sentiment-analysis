pip install joblib

import joblib

# Load the model
model = joblib.load('logistic_regression_model.pkl')

# Example usage: Predict with new data
data = [[feature1, feature2, feature3]]  # Example input data
prediction = model.predict(data)
print(prediction)
