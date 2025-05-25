# 🍷 Wine Quality Predictor

## Problem Statement

Predict the quality of wine based on its chemical properties, including alcohol content, acidity, and
pH levels, to help winemakers and consumers make informed decisions.

Wine quality assessment is important for producers and consumers to ensure product satisfaction and market competitiveness. Traditionally, wine quality is evaluated through expert tasting, which can be subjective, time-consuming, and costly.

This project aims to build a machine learning model to **predict the quality of red wine** based on its physicochemical properties (such as acidity, sugar content, pH, alcohol level, etc.). The goal is to classify whether a wine is of **good quality** or **not** with reasonable confidence, enabling faster and more objective quality control.

---

## Project Overview

A machine learning project that predicts whether a red wine sample is of **good quality** or **not** based on its physicochemical properties. The project includes:

- Data exploration and visualization
- Feature engineering and preprocessing
- Model training using a Random Forest classifier
- Model evaluation with accuracy, classification report, and confusion matrix
- Saving the trained model for future predictions
- A Streamlit web app for user-friendly wine quality prediction with confidence scores and feature comparison against ideal values

---

## 📂 Project Structure 
```
Wine-Quality-Predictor/
├── venv/ # Python virtual environment
├── ideal_ranges.csv # Min-max ideal feature ranges for good wines
├── ideal_values.csv # Mean ideal feature values for good quality wines
├── input.py # CLI version for wine quality prediction
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── wine_app.py # Streamlit web app for prediction
├── wine_model.pkl # Trained Random Forest model
├── wine_quality.py # EDA, model training, and evaluation
├── winequality-red.csv # Original dataset (UCI Wine Quality dataset)
```


---

## Installation and Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/ANJALI99-STACK/Wine-Quality-Predictor
    cd Wine-Quality-Predictor
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure you have packages like pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, etc.)*

3. Run the Streamlit app:
    ```bash
    streamlit run wine_app.py
    ```

---

## Usage

### Data Analysis & Model Training

Run the `wine_quality.py` script to:

- Explore the dataset
- Visualize feature distributions and correlations
- Convert wine quality to a binary target (`good_quality`)
- Train a Random Forest classifier
- Evaluate the model performance
- Save the trained model and ideal feature values for later use

### Predicting Wine Quality (Streamlit app)

1. Open the Streamlit app.
2. Enter wine features (e.g., acidity, sugar, pH, alcohol) within suggested ranges.
3. Click **Predict Quality**.
4. View the predicted wine quality class (Good / Not Good) with confidence.
5. See a detailed comparison of your input features against ideal values, with suggestions to improve quality.

---

## Sample Input Features

| Feature             | Typical Value Range   | Example Input |
|---------------------|----------------------|---------------|
| fixed acidity       | 4.0 – 16.0           | 7.5           |
| volatile acidity    | 0.1 – 1.5            | 0.4           |
| citric acid         | 0.0 – 1.0            | 0.3           |
| residual sugar      | 0.5 – 15.0           | 2.5           |
| chlorides           | 0.01 – 0.2           | 0.05          |
| free sulfur dioxide | 1 – 75               | 15            |
| total sulfur dioxide| 6 – 300              | 46            |
| density             | 0.9900 – 1.0050      | 0.9965        |
| pH                  | 2.5 – 4.5            | 3.3           |
| sulphates           | 0.2 – 1.5            | 0.6           |
| alcohol             | 8.0 – 15.0           | 10.5          |

---

## Model Performance

- Accuracy: Around 80% (varies with train-test split and hyperparameters)
- Good balance of precision and recall for the binary classification of good vs. not good wines

---

## Files Explanation

- **wine_quality.py** — Python script for EDA, training, evaluation, saving model and ideal values.
- **wine_app.py** — Streamlit web application for interactive prediction and feature analysis.
- **winequality-red.csv** — Dataset used for training.
- **wine_model.pkl** — Pickled trained Random Forest model.
- **ideal_values.csv** — Mean values of features for good quality wines.
- **ideal_ranges.csv** — Min and max ranges of features for good quality wines.

---

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- pickle (standard lib)

---

## License

This project is licensed under the [MIT License](LICENSE).


---

## Acknowledgments

- Dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- Thanks to the data contributors and community

---

Feel free to contribute or raise issues! 🍷😊
