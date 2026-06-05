import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Set visualization styles
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Load data
# Tell Python to look inside the subfolder
df = pd.read_csv('Online Shopping Intention Analysis/online_shoppers_intention.csv')

# 2. Basic Inspection
print("--- Dataset Shape ---")
print(df.shape)

print("\n--- Target Class Distribution (Imbalance Check) ---")
print(df['Revenue'].value_counts(normalize=True))

print("\n--- Missing Values Check ---")
print(df.isnull().sum().sum())

# Separate features (X) and target variable (y)
X = df.drop(columns=['Revenue'])
y = df['Revenue'].astype(int) # Convert True/False to 1/0

# Define feature categories
num_features = [
    'Administrative', 'Administrative_Duration', 
    'Informational', 'Informational_Duration', 
    'ProductRelated', 'ProductRelated_Duration', 
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]

cat_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']

# Create transformers for numerical and categorical data
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor block
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create an end-to-end Machine Learning Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
])

# Train the model
print("Training the Random Forest Model...")
model_pipeline.fit(X_train, y_train)
print("Training Complete!")

# Make predictions
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

# 1. Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 2. ROC-AUC Score
auc_score = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {auc_score:.4f}")

# 3. Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Purchase', 'Purchase'], yticklabels=['No Purchase', 'Purchase'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Extract feature names after One-Hot Encoding
encoded_cat_features = model_pipeline.named_steps['preprocessor'].transformers_[1][1]\
    .named_steps['onehot'].get_feature_names_out(cat_features)
all_features = num_features + list(encoded_cat_features)

# Extract importances from the classifier
importances = model_pipeline.named_steps['classifier'].feature_importances_

# Map and sort them
feature_imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})\
    .sort_values(by='Importance', ascending=False)

# Plot top 10 features
plt.figure(figsize=(10, 6))
# Fix for the Future Warning
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=feature_imp_df.head(10), 
    hue='Feature',          # Assigning y to hue satisfies the new syntax
    palette='viridis', 
    legend=False            # Hides the redundant legend box
)
plt.title('Top 10 Most Influential Features for Shopping Intention')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.show()

import joblib

# Save the entire pipeline
joblib.dump(model_pipeline, 'shopping_intent_model.pkl')
print("Model saved successfully as 'shopping_intent_model.pkl'")