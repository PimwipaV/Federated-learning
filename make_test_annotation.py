import os

# Directory containing image and text files
data_dir = '../data/DatasetVal'

# Create the output directory if it doesn't exist
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

output_file_path = '../consolidated_test_labels.txt'
with open(output_file_path, 'w') as keras_labels_file:

# Iterate through text files in the data directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            txt_path = os.path.join(data_dir, filename)
            image_path = os.path.join(data_dir, filename.replace('.txt', '.jpg'))

            with open(txt_path, 'r') as txt_file:
                # Read class_id, x_center, y_center, width, and height
                keras_label = ''
                class_id, x_center, y_center, width, height = [float(val) for val in txt_file.read().split()]
                
                keras_label += f'{image_path} {int(class_id)} {float(x_center)} {float(y_center)} {float(width)} {float(height)}\n'
          
                keras_labels_file.write(keras_label)
