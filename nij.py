import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Remove 'ID' and 'Training_Sample' columns
data = data.drop(columns=['ID', 'Training_Sample'])

# Convert all values to numeric, setting errors='coerce' to handle non-numeric values
data = data.apply(pd.to_numeric, errors='coerce')

# Fill missing values with the median of each column
data = data.fillna(data.median())

# Check for any remaining NaN values and fill them with 0
data = data.fillna(0)

# Ensure there are no NaN values left
assert data.isnull().sum().sum() == 0, "There are still NaN values in the dataset"

# Remove constant features, excluding important ones
important_features = ['Gender', 'Race', 'Age_at_Release', 'Supervision_Level_First', 'Education_Level', 'Prison_Offense', 'Prison_Years']
data = data.loc[:, (data != data.iloc[0]).any() | data.columns.isin(important_features)]

# Check variance of each feature and print those with zero or very low variance
variance_threshold = 0.01
low_variance_features = data.var()[data.var() < variance_threshold].index
print("Features with zero or low variance:", low_variance_features.tolist())

# Remove features with zero or low variance
data = data.drop(columns=low_variance_features)

# Define the target variable and features
target = 'Recidivism_Within_3years'
features = data.drop(columns=['Recidivism_Within_3years', 'Recidivism_Arrest_Year1', 'Recidivism_Arrest_Year2', 'Recidivism_Arrest_Year3']).columns

# Convert the target variable to categorical
data[target] = data[target].astype('category').cat.codes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=42)

# Initialize the Random Forest classifier with 493 trees
clf = RandomForestClassifier(n_estimators=493, random_state=42)

# Fit the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate prediction accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Prediction Accuracy: {accuracy:.4f}')

# Calculate chi-squared value and p-value
chi2_values, p_values = chi2(X_train, y_train)
chi2_df = pd.DataFrame({'Feature': features, 'Chi2': chi2_values, 'p-value': p_values})
chi2_df['Chi2'] = chi2_df['Chi2'].apply(lambda x: round(x, 5))
chi2_df['p-value'] = chi2_df['p-value'].apply(lambda x: round(x, 7))

# Sort features based on chi-squared values
chi2_df = chi2_df.sort_values(by='Chi2', ascending=False)
print(chi2_df)
chi2_df.to_csv('chi2_values.csv', index=False)

# Get feature importances
feature_importances = clf.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print top ten feature importances
print(importance_df.head(10))
importance_df.to_csv('feature_importances.csv', index=False)

