import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

np.random.seed(12)

normal_volume = np.random.negative_binomial(n=10, p=0.3, size=1000) + 1
normal_night_ratio = np.random.beta(a=1, b=5, size=1000)
criminal_volume = np.array([120, 140, 5, 95, 110])
criminal_night_ratio = np.array([0.85, 0.90, 0.95, 0.10, 0.75])

volumes = np.concatenate([normal_volume, criminal_volume])
night_ratios = np.concatenate([normal_night_ratio, criminal_night_ratio])

df = pd.DataFrame({
    'Caller_ID': [f'User_{i}' for i in range(len(volumes))],
    'Total_Calls_Per_Week': volumes,
    'Nighttime_Call_Ratio': night_ratios
})

model = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly_Flag'] = model.fit_predict(df[['Total_Calls_Per_Week', 'Nighttime_Call_Ratio']])

df['Status'] = df['Anomaly_Flag'].map({1: 'Normal Profile', -1: 'Suspect Profile'})

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Total_Calls_Per_Week',
    y='Nighttime_Call_Ratio',
    hue='Status',
    palette={'Normal Profile': '#1f77b4', 'Suspect Profile': '#d62728'},
    alpha=0.8
)

plt.title('Behavioral Anomaly Detection in Phone Call Records', fontsize=14, fontweight='bold')
plt.xlabel('Total Calls (Per Week)')
plt.ylabel('Ratio of Nighttime Calls (12 AM - 5 AM)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Model Classification')
plt.show()

print("Top Flagged Suspect Profiles:")
print(df[df['Anomaly_Flag'] == -1].to_string(index=False))
