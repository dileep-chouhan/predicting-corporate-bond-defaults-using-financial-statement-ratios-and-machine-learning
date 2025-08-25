import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_samples = 500
data = {
    'DebtToEquity': np.random.uniform(0.5, 2.5, num_samples),
    'CurrentRatio': np.random.uniform(0.8, 2.0, num_samples),
    'InterestCoverage': np.random.uniform(0.5, 5.0, num_samples),
    'ReturnOnAssets': np.random.uniform(-0.1, 0.2, num_samples),
    'Default': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]) # 20% default rate
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
# (In a real-world scenario, this would involve handling missing values, outliers, etc.)
# For this synthetic data, no cleaning is explicitly needed.
# --- 3. Model Training ---
X = df[['DebtToEquity', 'CurrentRatio', 'InterestCoverage', 'ReturnOnAssets']]
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(solver='liblinear') # Choosing a solver appropriate for smaller datasets.
model.fit(X_train, y_train)
# --- 4. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)
# --- 5. Visualization ---
# Example: Visualizing the relationship between DebtToEquity and Default
plt.figure(figsize=(8, 6))
sns.scatterplot(x='DebtToEquity', y='Default', data=df, hue='Default')
plt.title('Debt to Equity Ratio vs. Default')
plt.xlabel('Debt to Equity Ratio')
plt.ylabel('Default (0=No, 1=Yes)')
plt.savefig('debt_equity_vs_default.png')
print("Plot saved to debt_equity_vs_default.png")
#Example: Visualizing feature importance (if applicable to the model)
#This part is commented out because feature importance is not directly available for LogisticRegression in a readily interpretable way like it is for tree-based models.  To get feature importance you would need a different model (like RandomForest).
#feature_importance = model.feature_importances_
#plt.figure(figsize=(8,6))
#plt.bar(X.columns, feature_importance)
#plt.title('Feature Importance')
#plt.xlabel('Features')
#plt.ylabel('Importance')
#plt.savefig('feature_importance.png')
#print("Plot saved to feature_importance.png")