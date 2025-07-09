import numpy as np
import pickle
import pandas as pd

# Load model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

print("=== Student Final Score Predictor ===")
hours = float(input("Enter hours studied: "))
attendance = float(input("Enter attendance percentage: "))
prev_score = float(input("Enter previous exam score: "))
extra = int(input("Participates in extracurricular activities? (1 for Yes, 0 for No): "))

# Convert to DataFrame to include feature names
input_data = pd.DataFrame([{
    "hours_studied": hours,
    "attendance": attendance,
    "previous_score": prev_score,
    "extracurricular": extra
}])

pred = model.predict(input_data)
print(f"\nPredicted final score: {pred[0]:.2f}")
