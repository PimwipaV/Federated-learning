import os
import shutil

# Source directory
source_dir = '../data/Dataset'

# Destination directory
destination_dir = '../data/DatasetVal'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# List of filenames to be copied
#filenames_to_copy = ['file1.txt', 'file2.jpg', 'file3.png']

# Path to the file containing file paths
filepaths_file = '../data/test_images_and_labels.txt'

# Initialize the list to store filenames
filenames_to_copy = []

# Read the file and extract filenames
with open(filepaths_file, 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace and add to the list
        filenames_to_copy.append(line.strip())

# Now, you have a list of filenames to copy
print(filenames_to_copy)


# Iterate over the list of filenames
for filename in filenames_to_copy:
    # Construct the source and destination paths
    source_path = os.path.join(source_dir, filename)
    destination_path = os.path.join(destination_dir, filename)

    # Check if the source file exists
    #if os.path.exists(source_path):
        # Copy the file to the destination directory
    shutil.copy(source_path, destination_path)
    print(f"Copied: {filename} to {destination_dir}")
else:
    print(f"File not found: {filename}")

print("Copy process completed.")
