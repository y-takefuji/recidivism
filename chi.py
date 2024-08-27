import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the dataset
df = pd.read_csv('Recidivism__Beginning_2008.csv')

# Remove "UNKNOWN" from 'County of Indictment'
df = df[df['County of Indictment'] != 'UNKNOWN']

# Initialize lists to store chi-squared and p-values
chi2_county = []
p_county = []
chi2_gender = []
p_gender = []
chi2_age = []
p_age = []

# Loop through each year from 2008 to 2020
for year in range(2008, 2021):
    df_year = df[df['Release Year'] == year]
    
    # Chi-squared test for 'Return Status' and 'County of Indictment'
    contingency_table_county = pd.crosstab(df_year['Return Status'], df_year['County of Indictment'])
    chi2, p, _, _ = chi2_contingency(contingency_table_county)
    chi2_county.append(chi2)
    p_county.append(p)
    
    # Chi-squared test for 'Return Status' and 'Gender'
    contingency_table_gender = pd.crosstab(df_year['Return Status'], df_year['Gender'])
    chi2, p, _, _ = chi2_contingency(contingency_table_gender)
    chi2_gender.append(chi2)
    p_gender.append(p)
    
    # Chi-squared test for 'Return Status' and 'Age at Release'
    contingency_table_age = pd.crosstab(df_year['Return Status'], df_year['Age at Release'])
    chi2, p, _, _ = chi2_contingency(contingency_table_age)
    chi2_age.append(chi2)
    p_age.append(p)

# Plot the trends
years = list(range(2008, 2021))

fig, ax1 = plt.subplots(figsize=(14, 8))

# Chi-squared values on the left Y-axis
ax1.plot(years, chi2_county, label='Chi-squared (County)', linestyle='-', linewidth=1, color='black')
ax1.plot(years, chi2_gender, label='Chi-squared (Gender)', linestyle='--', linewidth=1, color='black')
ax1.plot(years, chi2_age, label='Chi-squared (Age)', linestyle='-.', linewidth=1, color='black')
ax1.set_xlabel('Year')
ax1.set_ylabel('Chi-squared Values')
ax1.tick_params(axis='y')

# P-values on the right Y-axis
ax2 = ax1.twinx()
ax2.plot(years, p_county, label='P-value (County)', linestyle='-', linewidth=2, color='gray')
ax2.plot(years, p_gender, label='P-value (Gender)', linestyle='--', linewidth=2, color='gray')
ax2.plot(years, p_age, label='P-value (Age)', linestyle='-.', linewidth=2, color='gray')
ax2.axhline(y=0.05, color='red', linestyle=':', linewidth=2, label='Reference Line (0.05)')
ax2.set_ylabel('P-values')
ax2.tick_params(axis='y')

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

plt.title('Trends of Chi-squared and P-values (2008-2020)')
plt.xticks(years, rotation=90)
plt.grid(True)
plt.tight_layout()
plt.savefig('chi.png',dpi=300)
plt.show()

