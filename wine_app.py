import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load ideal values from CSV (used for comparison)
ideal_values = pd.read_csv('ideal_values.csv', index_col=0)['Ideal']

# Load the trained model
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title and description
st.title("🍷 Wine Quality Predictor")
st.write("Enter the wine's properties to predict its quality and compare your input to ideal values.")

# Valid input ranges for each feature
feature_info = {
    "fixed acidity": (4.0, 16.0),
    "volatile acidity": (0.1, 1.5),
    "citric acid": (0.0, 1.0),
    "residual sugar": (0.5, 15.0),
    "chlorides": (0.01, 0.2),
    "free sulfur dioxide": (1, 75),
    "total sulfur dioxide": (6, 300),
    "density": (0.9900, 1.0050),
    "pH": (2.5, 4.5),
    "sulphates": (0.2, 1.5),
    "alcohol": (8.0, 15.0)
}

# Collect user inputs
inputs = {}
for feature, (min_val, max_val) in feature_info.items():
    val_str = st.text_input(f"{feature} (Range: {min_val} – {max_val})", key=feature)
    if val_str.strip() == "":
        inputs[feature] = None
    else:
        try:
            val = float(val_str)
            if not (min_val <= val <= max_val):
                st.error(f"❗ {feature} must be between {min_val} and {max_val}.")
                inputs[feature] = None
            else:
                inputs[feature] = val
        except ValueError:
            st.error(f"❗ Please enter a valid number for {feature}.")
            inputs[feature] = None

# Predict button logic
if st.button("Predict Quality"):
    if None in inputs.values():
        st.error("🚫 Please fill in all inputs correctly within the specified ranges.")
    else:
        # Convert input to array and make prediction
        feature_values = np.array([[inputs[feat] for feat in feature_info]])
        prediction = model.predict(feature_values)[0]
        prob = model.predict_proba(feature_values)[0]

        # Display prediction result with confidence
        if prediction == 1:
            st.success(f"✅ The wine is predicted to be GOOD quality. (Confidence: {prob[1] * 100:.2f}%)")
        else:
            st.error(f"❌ The wine is predicted to be NOT GOOD quality. (Confidence: {prob[0] * 100:.2f}%)")

        # -------- Feature Comparison Section --------
        st.write("### 📊 Feature Analysis Compared to Ideal Values")

        # Create user input series and calculate deviation
        user_series = pd.Series(feature_values[0], index=ideal_values.index)
        deviation = (user_series - ideal_values).abs()
        suggestions = []

        # Generate suggestions based on deviation from ideal
        for feat in ideal_values.index:
            user_val = user_series[feat]
            ideal_val = ideal_values[feat]
            if deviation[feat] > 0.1 * ideal_val:
                direction = "higher" if user_val < ideal_val else "lower"
                suggestions.append(f"Should be {direction}")
            else:
                suggestions.append("✅ Acceptable")

        # Create a comparison DataFrame
        comparison_df = pd.DataFrame({
            "Your Input": user_series,
            "Ideal Value": ideal_values,
            "Deviation": deviation,
            "Suggestion": suggestions
        }).sort_values(by="Deviation", ascending=False)

        # Highlight problematic features in red
        def highlight_deviation(row):
            return ['background-color: #ffcccc' if row['Suggestion'].startswith("Should") and col == 'Your Input' else '' for col in row.index]

        # Display the comparison table
        st.dataframe(comparison_df.style.apply(highlight_deviation, axis=1))
