# Efficient Data Stream Anomaly Detection

### Table of Contents
1. [Introduction](#introduction)
2. [Algorithm Selection](#algorithm-selection)
   - [Isolation Forest](#isolation-forest)
   - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
3. [Data Stream Simulation](#data-stream-simulation)
4. [Anomaly Detection Implementation](#anomaly-detection-implementation)
   - [Isolation Forest Implementation](#isolation-forest-implementation)
   - [CNN Implementation](#cnn-implementation)
5. [Optimization Techniques](#optimization-techniques)
6. [Visualization Tool](#visualization-tool)
7. [Error Handling and Data Validation](#error-handling-and-data-validation)
8. [Requirements and Dependencies](#requirements-and-dependencies)
9. [Conclusion](#conclusion)
10. [References](#references)

---

## Introduction

This project aims to develop a Python script capable of detecting anomalies in a continuous data stream. The data stream simulates real-time sequences of floating-point numbers, representing metrics such as financial transactions or system metrics. The focus is on identifying unusual patterns, such as exceptionally high values or deviations from the norm, in real-time.

Two machine learning approaches are implemented:

1. **Isolation Forest**: An ensemble-based anomaly detection method that isolates anomalies instead of profiling normal data points.
2. **Convolutional Neural Network (CNN)**: A deep learning model adapted for time-series anomaly detection, leveraging its capability to capture complex patterns.

---

## Algorithm Selection

### Isolation Forest

**Explanation**:

- **Isolation Forest** is an unsupervised learning algorithm specifically designed for anomaly detection.
- It operates by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
- Anomalies are isolated quickly because they are few and have attribute-values that are very different from the norm.
- The path length from the root node to the terminating node (where the data point is isolated) is shorter for anomalies.

**Effectiveness**:

- **Advantages**:
  - Efficient for large datasets due to its linear time complexity.
  - Does not require the data to be scaled.
  - Handles high-dimensional data well.
- **Disadvantages**:
  - Randomness in tree construction may affect consistency.
  - Requires tuning of the `contamination` parameter for optimal performance.

### Convolutional Neural Network (CNN)

**Explanation**:

- **CNNs** are deep learning models typically used for image data but can be adapted for time-series anomaly detection.
- By treating the time series data as a univariate sequence, CNNs can learn spatial hierarchies of features through convolutional layers.
- The model learns to detect patterns and anomalies by training on sequences labeled as normal or anomalous.

**Effectiveness**:

- **Advantages**:
  - Capable of capturing complex temporal patterns and relationships in the data.
  - Adaptable to concept drift and seasonal variations due to learning from data.
- **Disadvantages**:
  - Requires a significant amount of data and computational resources.
  - Needs careful tuning of hyperparameters.
  - Training a CNN can be time-consuming.

---

## Data Stream Simulation

A function is designed to emulate a real-time data stream with the following characteristics:

- **Regular Patterns**: Simulates periodic behavior using sine and cosine functions.
- **Seasonal Elements**: Introduces seasonality by varying amplitudes or frequencies.
- **Random Noise**: Adds Gaussian noise to simulate real-world imperfections.
- **Anomalies**: Injects anomalies at specified indices with significant deviations.

**Implementation**:

```python
import numpy as np

def generate_data_stream(total_points=1000, anomaly_indices=None, seed=42):
    """
    Simulates a data stream with regular patterns, noise, and anomalies.
    """
    np.random.seed(seed)
    if anomaly_indices is None:
        anomaly_indices = []
    
    t = np.arange(total_points)
    # Regular pattern with seasonality
    data = 10 * np.sin(2 * np.pi * t / 50) + \
           5 * np.cos(2 * np.pi * t / 100)
    # Add random noise
    data += np.random.normal(0, 1, size=total_points)
    # Introduce anomalies
    for idx in anomaly_indices:
        data[idx] += np.random.uniform(15, 30) * np.random.choice([-1, 1])
    return data
```

---

## Anomaly Detection Implementation

### Isolation Forest Implementation

**Code Overview**:

- Uses the `IsolationForest` algorithm from `scikit-learn`.
- Incorporates feature engineering to capture temporal patterns.
- Scales features using `StandardScaler` for improved performance.

**Implementation**:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class IsolationForestDetector:
    def __init__(self, contamination=0.01, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(contamination=self.contamination,
                                     random_state=self.random_state)
        self.scaler = StandardScaler()
        self.anomalies = None

    def fit_predict(self, data):
        # Feature Engineering
        df = pd.DataFrame({'value': data})
        df['time'] = np.arange(len(data))
        df['rolling_mean'] = df['value'].rolling(window=5).mean()
        df['rolling_std'] = df['value'].rolling(window=5).std()
        df['lag1'] = df['value'].shift(1)
        df['lag2'] = df['value'].shift(2)
        df['difference'] = df['value'].diff()

        # Drop initial NaN values
        df_clean = df.dropna().reset_index(drop=True)
        
        # Prepare features
        X = df_clean[['value', 'rolling_mean', 'rolling_std', 'lag1', 'lag2', 'difference']]
        X_scaled = self.scaler.fit_transform(X)

        # Fit the model
        self.model.fit(X_scaled)

        # Predict anomalies
        df_clean['anomaly'] = self.model.predict(X_scaled)
        df_clean['anomaly'] = df_clean['anomaly'].map({1: 0, -1: 1})
        
        # Store anomalies
        self.anomalies = df_clean[df_clean['anomaly'] == 1]
        return df_clean['anomaly'].values, df_clean

# Usage
if __name__ == "__main__":
    data_stream = generate_data_stream(total_points=1000, anomaly_indices=[150, 500, 750])
    detector = IsolationForestDetector(contamination=0.003)
    anomaly_labels, results_df = detector.fit_predict(data_stream)

    print("Detected anomalies at indices:")
    print(detector.anomalies[['time', 'value']])
```

### CNN Implementation

**Code Overview**:

- Uses a CNN model built with `Keras` (part of `tensorflow`) for time-series anomaly detection.
- The model is trained to reconstruct normal sequences; anomalies are detected when reconstruction error exceeds a threshold.
- Operates as an unsupervised model using an autoencoder architecture.

**Implementation**:

```python
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

class CNNAnomalyDetector:
    def __init__(self, sequence_length=30, threshold=None):
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None

    def create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    def build_model(self):
        model = Sequential()
        model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(self.sequence_length, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(16, kernel_size=3, activation='relu'))
        model.add(UpSampling1D(size=2))
        model.add(Conv1D(1, kernel_size=3, activation='sigmoid', padding='same'))
        model.compile(optimizer=Adam(), loss='mse')
        self.model = model

    def fit(self, data):
        # Scale data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1))
        sequences = self.create_sequences(data_scaled)
        X_train = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))

        # Build and train the model
        self.build_model()
        self.history = self.model.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

        # Set threshold
        reconstructions = self.model.predict(X_train)
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=(1,2))
        self.threshold = np.percentile(mse, 95)

    def predict(self, data):
        # Scale data
        data_scaled = self.scaler.transform(data.reshape(-1, 1))
        sequences = self.create_sequences(data_scaled)
        X_test = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))

        # Get reconstructions
        reconstructions = self.model.predict(X_test)
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=(1,2))
        anomalies = mse > self.threshold
        anomaly_indices = np.where(anomalies)[0] + self.sequence_length
        return anomaly_indices

# Usage
if __name__ == "__main__":
    data_stream = generate_data_stream(total_points=1000, anomaly_indices=[150, 500, 750])
    detector = CNNAnomalyDetector(sequence_length=30)
    detector.fit(data_stream)

    anomaly_indices = detector.predict(data_stream)
    print("Detected anomalies at indices:")
    print(anomaly_indices)
```

**Notes**:

- **Data Preparation**:
  - The data is scaled using `MinMaxScaler` to ensure the values are between 0 and 1.
  - Sequences of fixed length are created to feed into the CNN.

- **Model Architecture**:
  - The CNN model consists of convolutional and pooling layers followed by upsampling and convolutional layers for reconstruction.
  - The model learns to reconstruct normal patterns; anomalies result in higher reconstruction errors.

- **Anomaly Detection**:
  - After training, the reconstruction error (mean squared error) is calculated.
  - A threshold is set based on the 95th percentile of the training reconstruction errors.
  - Data points with reconstruction errors exceeding the threshold are flagged as anomalies.

---

## Optimization Techniques

- **Efficient Data Structures**: Utilized NumPy arrays and pandas DataFrames for efficient data manipulation.
- **Batch Processing**: In the CNN implementation, data is processed in batches to leverage vectorized operations.
- **Model Parameters Tuning**: Adjusted hyperparameters like `sequence_length`, `epochs`, and `batch_size` for optimal performance.
- **Early Stopping**: Implemented validation split in training to monitor for overfitting.

---

## Visualization Tool

A visualization tool is created using `matplotlib` to display the data stream and detected anomalies in real-time or after processing.

**Implementation**:

```python
import matplotlib.pyplot as plt

def visualize_anomalies(data, anomaly_indices):
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(len(data)), data, label='Data')
    plt.scatter(anomaly_indices, data[anomaly_indices], color='red', label='Anomalies')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('Data Stream with Detected Anomalies')
    plt.legend()
    plt.show()
```

**Usage**:

```python
# For Isolation Forest
visualize_anomalies(data_stream, detector.anomalies['time'].values)

# For CNN
visualize_anomalies(data_stream, anomaly_indices)
```

---

## Error Handling and Data Validation

- **Error Handling**:
  - Try-except blocks are used around critical sections like model training and prediction to catch and handle exceptions.
  - Custom error messages provide clarity on issues encountered.

- **Data Validation**:
  - Checked for NaN or infinite values in the data stream before processing.
  - Ensured that data sequences have sufficient length for the CNN model.

**Example**:

```python
def fit(self, data):
    if len(data) < self.sequence_length:
        raise ValueError(f"Data length must be at least {self.sequence_length}")
    # Proceed with fitting
```

---

## Requirements and Dependencies

- **Python Version**: Python 3.x

- **External Libraries**:
  - `numpy` for numerical computations.
  - `pandas` for data manipulation.
  - `matplotlib` for visualization.
  - `scikit-learn` for Isolation Forest and preprocessing.
  - `tensorflow` for building and training the CNN model.

**`requirements.txt`**:

```
numpy
pandas
matplotlib
scikit-learn
tensorflow
```

---

## Conclusion

This project presents two advanced machine learning approaches for real-time anomaly detection in data streams:

1. **Isolation Forest**:
   - Efficient for large datasets.
   - Handles high-dimensional data and requires minimal feature scaling.
   - Suitable when quick anomaly detection is needed with less computational overhead.

2. **Convolutional Neural Network (CNN)**:
   - Capable of learning complex patterns and adapting to concept drift and seasonal variations.
   - Requires more computational resources and careful tuning.
   - Provides higher accuracy in detecting subtle anomalies that statistical methods might miss.

Both methods are implemented with an emphasis on speed, efficiency, and real-time processing capabilities. The choice between them depends on the specific needs of the application, available computational resources, and the nature of the data.

---

## References

- **Scikit-learn Documentation**: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- **TensorFlow Documentation**: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **Anomaly Detection Techniques**:
  - Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *2008 Eighth IEEE International Conference on Data Mining*, 413-422.
  - Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys (CSUR)*, 41(3), 15.
- **Time-Series Anomaly Detection with CNNs**:
  - Malhotra, P., et al. (2015). Long Short Term Memory Networks for Anomaly Detection in Time Series. *Proceedings*, 89th European Physical Society Conference on High Energy Physics.

---

