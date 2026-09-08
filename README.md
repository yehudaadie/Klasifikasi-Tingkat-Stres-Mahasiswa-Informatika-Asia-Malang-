# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import joblib

sns.set(style="whitegrid")

print("Library loaded successfully!")

# ==========================================
# 2. LOAD DATASET
# ==========================================
from google.colab import files

print("Upload dataset StressLevelDataset.csv")
uploaded = files.upload()

df = pd.read_csv(next(iter(uploaded.keys())))
df.head()

# ==========================================
# 3. CEK STRUKTUR DATA
# ==========================================
print("Shape:", df.shape)
print("\nInfo Data:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

df.describe()

# ==========================================
# 4. DISTRIBUSI LABEL (EDA)
# ==========================================
plt.figure(figsize=(6,4))
sns.countplot(x=df['stress_level'])
plt.title("Distribusi Kelas Stress Level")
plt.show()
df['stress_level'].value_counts()

# ==========================================
# 5. KORELASI FITUR (EDA)
# ==========================================
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Heatmap Korelasi Fitur")
plt.show()

# ==========================================
# 6. SPLIT DATASET
# ==========================================
X = df.drop('stress_level', axis=1)
y = df['stress_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ==========================================
# 7. TRAIN RANDOM FOREST (MODEL DASAR)
# ==========================================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Akurasi Model Dasar:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 9. HYPERPARAMETER TUNING
# ==========================================
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2'],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)

# ==========================================
# 10. EVALUASI MODEL HASIL TUNING
# ==========================================
best_rf = grid.best_estimator_

y_pred_best = best_rf.predict(X_test)

print("Akurasi (Best Model):", accuracy_score(y_test, y_pred_best))
print("\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred_best))

# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10,6))
importances.head(15).plot(kind='bar')
plt.title("Top 15 Feature Importance - Random Forest")
plt.xlabel("Fitur")
plt.ylabel("Pentingnya")
plt.show()

importances
