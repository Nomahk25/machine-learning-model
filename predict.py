import pickle
import numpy as np

with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

sample = np.array([[6, 85, 80, 1]])

prediction = model.predict(sample)
print(f"Predicted final score: {prediction[0]:.2f}")
