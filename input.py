import pickle
import numpy as np
import pandas as pd

# Load the trained wine quality prediction model
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the feature names required by the model
features = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

# Collect user input for each feature
user_input = []
print("🍷 Enter the following wine properties:")
for feature in features:
    while True:
        try:
            value = float(input(f"{feature}: "))
            user_input.append(value)
            break
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")

# Convert the input to a NumPy array for model prediction
user_array = np.array([user_input])

# Predict wine quality (0 = Not Good, 1 = Good)
prediction = model.predict(user_array)
prob = model.predict_proba(user_array)[0]  # Probabilities for both classes

# Display prediction result with confidence
if prediction[0] == 1:
    print(f"\n✅ The wine is predicted to be GOOD quality. (Confidence: {prob[1] * 100:.2f}%)")
else:
    print(f"\n❌ The wine is predicted to be NOT GOOD quality. (Confidence: {prob[0] * 100:.2f}%)")

    # Load ideal values for comparison (from a CSV file)
    ideal_values = pd.read_csv("ideal_values.csv", index_col=0)['Ideal']

    # Provide possible reasons for poor quality prediction
    print("\n⚠️ Likely reasons for NOT GOOD prediction:")
    for feat, val in zip(features, user_input):
        ideal = ideal_values[feat]
        if abs(val - ideal) > 0.1 * ideal:
            direction = "higher" if val < ideal else "lower"
            print(f" - {feat.capitalize()} is too {'low' if direction == 'higher' else 'high'} "
                  f"(Your: {val}, Ideal: {ideal:.2f}) → should be {direction}.")
