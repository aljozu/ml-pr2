from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import dataset as dataset
from knn import KNearestNeighbors
from decisionTree import DecisionTree
import numpy as np
# Load data
ds = dataset.data_set()
train_images, train_labels, test_images, test_labels = ds.get_data()

# Extract wavelet features
X_train_wavelet = ds.get_wavelet_features(train_images)
X_test_wavelet = ds.get_wavelet_features(test_images)

# Normalize features
scaler = StandardScaler()
X_train_wavelet = scaler.fit_transform(X_train_wavelet)
X_test_wavelet = scaler.transform(X_test_wavelet)
op = 2
if(op == 1):
    # Create k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    kx = KNearestNeighbors()
    # Train the classifier
    knn.fit(X_train_wavelet, train_labels)
    kx.fit(X_train_wavelet, train_labels)
    # Predict on test data
    predictions = knn.predict(X_test_wavelet)
    predictionsx = kx.predict(X_test_wavelet)
    # Evaluate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    accuracy2 = accuracy_score(test_labels, predictionsx)
    print(f"Accuracy with wavelet features: {accuracy:.4f}")
    print(f"Accuracy2 with wavelet features: {accuracy2:.4f}")
elif(op==2):
    # Create and train the decision tree
    tree = DecisionTree(max_depth=10, criterion="gini")
    tree.fit(X_train_wavelet, train_labels)

    # Predict the labels for new data points
    predictions = tree.predict(X_test_wavelet)
    print(predictions)  
