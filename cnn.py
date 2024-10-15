import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler

# Load your data from 'new_sample_data.csv'
data = pd.read_csv('sample_data.csv')

# Create time_steps feature
time_steps = np.arange(len(data)).reshape(-1, 1)

# Combine features
X = data['value'].values.reshape(-1, 1)
X_features = np.hstack((time_steps, X))

# Scale features
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features)

# Set contamination
outlier_fraction = 0.01  # 2 anomalies out of 100 data points

# Initialize KNN with increased n_neighbors
clf = KNN(contamination=outlier_fraction, n_neighbors=10)

# Fit the model
clf.fit(X_features_scaled)

# Get predictions
y_pred = clf.labels_
y_scores = clf.decision_scores_

# Add results to DataFrame
data['Anomaly'] = y_pred
data['Anomaly_Score'] = y_scores

# Print anomalies
anomalies = data[data['Anomaly'] == 1]
print("Detected anomalies:")
print(anomalies)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['value'], label='Data')
plt.scatter(anomalies.index, anomalies['value'], color='red', label='Anomalies')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('Anomaly Detection using KNN')
plt.legend()
plt.show()

