import numpy as np
import os
import struct
import pywt
import matplotlib.pyplot as plt

# File paths
data_dir = "./mnist_data"
train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")
test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")

class data_set:

    def __init__(self):
        pass

    # Function to read images
    def load_images(self, file_path):
        with open(file_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return images

    # Function to read labels
    def load_labels(self, file_path):
        with open(file_path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        return labels
    
    def get_data(self):
        train_images = self.load_images(train_images_path)
        train_labels = self.load_labels(train_labels_path)
        test_images = self.load_images(test_images_path)
        test_labels = self.load_labels(test_labels_path)

        return train_images, train_labels, test_images, test_labels
    
    def visualize_image_index(self, image_index, dataset='train'):
        if dataset == 'train':
            images, labels, _, _ = self.get_data()
        elif dataset == 'test':
            _, _, images, labels = self.get_data()
        else:
            raise ValueError("Invalid dataset. Choose 'train' or 'test'.")

        # Ensure the index is valid
        if image_index < 0 or image_index >= len(images):
            raise IndexError("Image index out of bounds.")
        
        # Get the image and label
        image = images[image_index]
        label = labels[image_index]

        # Plot the image
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')  # Hide the axes
        plt.show()

    def visualize_image_v(self, image_v):
        # Plot the image
        plt.imshow(image_v, cmap='gray')
        plt.axis('off')  # Hide the axes
        plt.show()

    def wavelet_transform(self, image):
        coeffs = pywt.dwt2(image, 'haar')  # Using Haar wavelet
        cA, (cH, cV, cD) = coeffs
        return np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])

    def get_wavelet_features(self, images):
        features = np.array([self.wavelet_transform(image) for image in images])
        return features




