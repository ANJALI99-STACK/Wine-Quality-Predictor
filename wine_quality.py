import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------
# 1. Load and Explore Dataset
# -----------------------------------
df = pd.read_csv("winequality-red.csv", sep=';')

# Preview data
print("First few rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())

# Plot distribution of wine quality
sns.countplot(x='quality', data=df)
plt.title('Distribution of Wine Quality Ratings')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# -----------------------------------
# 2. Feature Engineering
# -----------------------------------
# Convert quality into binary: good (>=6) → 1, else → 0
df['good_quality'] = df['quality'].apply(lambda q: 1 if q >= 6 else 0)

# Define feature columns (exclude quality and good_quality)
feature_columns = [col for col in df.columns if col not in ['quality', 'good_quality']]

# -----------------------------------
# 3. Generate Ideal Ranges and Values
# -----------------------------------
# Filter good quality wines
good_wines = df[df['good_quality'] == 1]

# Save ideal min/max range for each feature
ideal_ranges = good_wines[feature_columns].describe().loc[['min', 'max']].T
ideal_ranges.columns = ['Ideal Min', 'Ideal Max']
ideal_ranges.to_csv('ideal_ranges.csv')
print("✅ Ideal ranges saved to 'ideal_ranges.csv'")

# Save ideal mean values
ideal_values = good_wines[feature_columns].mean().to_frame(name='Ideal')
ideal_values.to_csv('ideal_values.csv')
print("✅ Ideal values saved to 'ideal_values.csv'")

# -----------------------------------
# 4. Train/Test Split
# -----------------------------------
X = df[feature_columns]
y = df['good_quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------
# 5. Model Training and Evaluation
# -----------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔢 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------------
# 6. Feature Importance Visualization
# -----------------------------------
feature_importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# -----------------------------------
# 7. Save the Model
# -----------------------------------
with open("wine_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved as 'wine_model.pkl'")

