import os
import cv2  # OpenCV for image processing
import numpy as np

# Define constants
image_dir = 'data/Dataset'
image_size = (32, 32)  # Target image size
split_ratio = 0.8  # 80% for training, 20% for testing

# Initialize lists to store data
x_data = []
y_data = []

# Load and process the data
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_dir, filename)
        txt_path = os.path.join(image_dir, filename.replace('.jpg', '.txt'))
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
        image = image / 255.0  # Normalize pixel values
        x_data.append(image)
        
        # Parse annotations from the text file
        with open(txt_path, 'r') as txt_file:
            # Parse and process the annotations (e.g., bounding box coordinates and class labels)
            class_id, x_center, y_center, width, height = [float(val) for val in txt_file.read().split()]
            yolo_label = [class_id, x_center, y_center, width, height]
            y_data.append(yolo_label)

# Convert lists to NumPy arrays
x_data = np.array(x_data)
y_data = np.array(y_data)

# Split the data into training and testing sets
split_index = int(len(x_data) * split_ratio)
x_train, x_test = x_data[:split_index], x_data[split_index:]
y_train, y_test = y_data[:split_index], y_data[split_index:]

print(len(x_data))
print(len(y_data))
#print(x_train)
print(x_train.shape)
print(y_train.shape)
print(y_train)
print(y_data)

