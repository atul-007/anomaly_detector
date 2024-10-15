import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load your data
data = pd.read_csv('sample_data.csv')

# Feature Engineering
data['rolling_mean'] = data['value'].rolling(window=5).mean()
data['rolling_std'] = data['value'].rolling(window=5).std()
data['difference'] = data['value'].diff()

# Drop initial NaN values resulting from feature engineering
data_clean = data.dropna().copy()  # Copy to avoid SettingWithCopyWarning

# Prepare feature set without lag features
X = data_clean[['value', 'rolling_mean', 'rolling_std', 'difference']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set contamination parameter
outlier_fraction = 2 / len(data_clean)

# Initialize Isolation Forest
model = IsolationForest(contamination=outlier_fraction, random_state=42)
model.fit(X_scaled)

# Predict anomalies
data_clean['anomaly'] = model.predict(X_scaled)
data_clean['anomaly'] = data_clean['anomaly'].map({1: 0, -1: 1})

# Get the anomalies
anomalies = data_clean[data_clean['anomaly'] == 1]

# Print detected anomalies
print("Detected anomalies:")
print(anomalies[['value', 'anomaly']])

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['value'], label='Data')
plt.scatter(anomalies.index, anomalies['value'], color='red', label='Anomalies')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Anomaly Detection using Isolation Forest')
plt.legend()
plt.show()
