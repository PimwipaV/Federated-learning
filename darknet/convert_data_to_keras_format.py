import os

dataset_dir = 'data/Dataset'
keras_labels_dir = 'data/keras_labels'

for filename in os.listdir(dataset_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(dataset_dir, filename), 'r') as darknet_file:
            lines = darknet_file.readlines()
            keras_label = ''
            for line in lines:
                class_id, x_center, y_center, width, height = line.strip().split()
                keras_label += f'{int(class_id)} {float(x_center)} {float(y_center)} {float(width)} {float(height)}\n'

        with open(os.path.join(keras_labels_dir, filename), 'w') as keras_file:
            keras_file.write(keras_label)