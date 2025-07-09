import pickle
import numpy as np

# Load model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input: hours_studied, attendance, previous_score, extracurricular
sample = np.array([[6, 85, 80, 1]])

# Predict
prediction = model.predict(sample)
print(f"Predicted final score: {prediction[0]:.2f}")
