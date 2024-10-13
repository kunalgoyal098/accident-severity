import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
data = pd.read_csv('path_to_accident_data.csv')
# Preprocess and clean data (handle missing values, convert categorical variables, etc.)

# Extract features and target
X = data[['feature1', 'feature2', ...]]  # Select relevant features
y = data['severity']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform Dimensionality Reduction (PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)

# Perform Clustering (KMeans)
kmeans = KMeans(n_clusters=3)
X_train_clusters = kmeans.fit_predict(X_train_scaled)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_pca, y_train)

# Evaluate the model
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)
y_pred = rf_classifier.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
