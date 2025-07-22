# Student Performance Predictor

This project predicts a student's final score based on their study habits, attendance, and previous performance.

## 🧠 Features
- Train a machine learning model (RandomForest)
- Predict student final score
- Analyze data visually (notebook)

## 📁 Structure
- `train.py` – trains and saves the model
- `predict.py` – uses the model to make predictions
- `main.py` – interactive input for predictions

## ✅ How to Run

### 1. Install dependencies:

```
pip install -r requirements.txt
```

### 2. Train the model:

```
python train.py
```

### 3. Predict:

```
python predict.py
```

### 4. Try the interactive app:

```
python main.py
```

## 🔍 Dataset

You can use your own CSV containing features like:
Hours studied
Attendance percentage
Previous exam scores
Participation level

## 📊 Model Info

Algorithm: Random Forest Regressor
Input Features: Configurable in train.py
Output: Predicted final score (e.g., 0–100)

## 🤝 Contributing

Feel free to fork the project, improve it, and submit pull requests!

Ideas:
- Add more ML algorithms
- Build a frontend UI using Flask or Streamlit
- Include support for other types of educational data

## 📄 License
This project is licensed under the MIT License.
Feel free to use it for educational or commercial purposes with proper credit.

## 👨‍💻 Author
Nomanguni
🌍 Johannesburg, South Africa
📫 GitHub | ✉️ nomanguni22@gmail.com

## 💡 Inspiration
Created to help students and educators understand how learning behaviors impact final performance through machine learning and simple predictive tools.
