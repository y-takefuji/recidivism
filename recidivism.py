import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('Recidivism__Beginning_2008.csv')

# Data preprocessing
df = df[df['County of Indictment'] != 'UNKNOWN']
df['Gender'] = df['Gender'].map({'MALE': 1, 'FEMALE': 0})

# Encode categorical variables
le_county = LabelEncoder()
df['County of Indictment'] = le_county.fit_transform(df['County of Indictment'])

le_return_status = LabelEncoder()
df['Return Status'] = le_return_status.fit_transform(df['Return Status'])

# Initialize lists to store Chi-squared values, p-values, and accuracy values
chi2_values = []
p_values = []
accuracy_values = []

# Calculate Chi-squared statistics, p-values, and accuracy values for each year
for year in range(2008, 2021):
    df_year = df[df['Release Year'] == year]
    X = df_year[['County of Indictment', 'Gender', 'Age at Release']]
    y = df_year['Return Status']
    chi2_val, p_val = chi2(X, y)
    chi2_values.append(chi2_val[0])
    p_values.append(p_val[0])
    
    # Train RandomForest model and calculate accuracy value
    model = RandomForestClassifier()
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    accuracy_values.append(accuracy)

# Convert lists to arrays for plotting
chi2_values = np.array(chi2_values)
p_values = np.array(p_values)
accuracy_values = np.array(accuracy_values)

# Plot Chi-squared values, p-values, and accuracy values
fig, ax1 = plt.subplots()

ax1.set_xlabel('Year')
ax1.set_ylabel('p-value', color='black')
ax1.plot(range(2008, 2021), p_values, color='black', linestyle='--', label='p-value')
ax1.tick_params(axis='y', labelcolor='black')
ax1.axhline(y=0.05, color='gray', linestyle=':', label='0.05 reference line')

ax2 = ax1.twinx()
ax2.set_ylabel('Chi-squared value', color='black')
ax2.plot(range(2008, 2021), chi2_values, color='black', linestyle='-', label='Chi-squared value')
ax2.tick_params(axis='y', labelcolor='black')

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel('Accuracy', color='black')
ax3.plot(range(2008, 2021), accuracy_values, color='black', linestyle=':', label='Accuracy')
ax3.tick_params(axis='y', labelcolor='black')

fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
plt.xticks(rotation=90)
plt.title('Chi-squared values, p-values, and Accuracy from 2008 to 2020')
plt.savefig('chi-p-acc.png',dpi=300)
plt.show()

# Train RandomForest model
X = df[['Release Year', 'County of Indictment', 'Gender', 'Age at Release']]
y = df['Return Status']
model = RandomForestClassifier()
model.fit(X, y)

# Calculate feature importances for each year
feature_importances = {feature: [] for feature in ['Release Year', 'County of Indictment', 'Gender', 'Age at Release']}
for year in range(2008, 2021):
    df_year = df[df['Release Year'] == year]
    X_year = df_year[['Release Year', 'County of Indictment', 'Gender', 'Age at Release']]
    y_year = df_year['Return Status']
    model.fit(X_year, y_year)
    importances = model.feature_importances_
    for i, feature in enumerate(feature_importances.keys()):
        feature_importances[feature].append(importances[i])

# Plot feature importances
plt.figure(figsize=(12, 8))
linestyles = ['-', '--', '-.', ':']
widths = [1, 2, 1, 2]
for (feature, linestyle, width) in zip(feature_importances.keys(), linestyles, widths):
    plt.plot(range(2008, 2021), feature_importances[feature], linestyle=linestyle, linewidth=width, label=feature, color='black')

plt.xlabel('Year')
plt.ylabel('Feature Importance')
plt.title('Feature Importances from 2008 to 2020')
plt.legend()
plt.xticks(range(2008, 2021), rotation=90)
plt.savefig('feature-importances.png',dpi=300)
plt.show()

