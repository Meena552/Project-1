# Project-1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Replace 'heart.csv' with your uploaded file name
dataset = pd.read_csv('heart.csv')
dataset.head()

print(dataset.isnull().sum())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in ['cp', 'restecg', 'slope', 'thal']:
    dataset[col] = le.fit_transform(dataset[col])

sns.countplot(x='target', data=dataset)
plt.title("Heart Disease Count (1 = Disease, 0 = No Disease)")
plt.show()

X = dataset.drop('target', axis=1)
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X = dataset.drop('target', axis=1)
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance - Random Forest")
plt.show()

# Group by age and sex, calculate average heart disease chance
risk_data = dataset.groupby(['age', 'sex'])['target'].mean().reset_index()

# Find the top highest risk group
highest_risk = risk_data.sort_values(by='target', ascending=False).iloc[0]

age = int(highest_risk['age'])
sex = 'Male' if highest_risk['sex'] == 1 else 'Female'
risk_percent = round(highest_risk['target'] * 100, 2)

print(f"🔍 Highest Risk Group:")
print(f"Age         : {age}")
print(f"Sex         : {sex}")
print(f"Risk Chance : {risk_percent}%")
