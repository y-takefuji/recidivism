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


# Train RandomForest model
X = df[['County of Indictment', 'Gender', 'Age at Release']]
y = df['Return Status']
model = RandomForestClassifier()
model.fit(X, y)

# Calculate feature importances for each year
feature_importances = {feature: [] for feature in ['County of Indictment', 'Gender', 'Age at Release']}
for year in range(2008, 2021):
    df_year = df[df['Release Year'] == year]
    X_year = df_year[['County of Indictment', 'Gender', 'Age at Release']]
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
plt.xticks(range(2008, 2021))
plt.savefig('feature-importances.png',dpi=300)
plt.show()

