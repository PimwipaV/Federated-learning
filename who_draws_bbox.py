########### Testing images on Yolov4-tiny ###############

def adjust_bounding_boxes(detections, image_width, image_height):
    adjusted_boxes = []
    
    for box in detections:
        x, y, width, height = box
        
        # Calculate the center of the bounding box
        center_x = x + (width / 2)
        center_y = y + (height / 2)
        
        # Calculate the offset to center the box
        offset_x = (width / 2) - (image_width / 2)
        offset_y = (height / 2) - (image_height / 2)
        
        # Adjust the bounding box coordinates
        adjusted_x = int(center_x - offset_x)
        adjusted_y = int(center_y - offset_y)
        adjusted_width = int(width)
        adjusted_height = int(height)
        
        adjusted_boxes.append([adjusted_x, adjusted_y, adjusted_width, adjusted_height])
    
    return adjusted_boxes


# Load the names of the objects
with open("obj.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the YOLOv4 Tiny model and its configuration
model = cv2.dnn.readNet("yolov4-tiny-custom_last.weights", "yolov4-tiny-custom.cfg")

# Load the input image
image = cv2.imread("test2.jpg")

# Create a blob from the image and set it as the input to the network
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
model.setInput(blob)

# Perform object detection and get the output
output_layers = model.getUnconnectedOutLayersNames()
layer_outputs = model.forward(output_layers)

# Loop over each output layer
for output in layer_outputs:
    # Loop over each detection
    for detection in output:
        # Extract the class ID and confidence
        scores = detection[5:]
        class_id = scores.argmax()
        confidence = scores[class_id]

        # Check if the confidence is above a threshold (e.g. 0.5)
        if confidence > 0.5:
            # Get the bounding box coordinates and scale them to the image size
            box = detection[0:4] * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
            x, y, w, h = box.astype(int)
            
            # # Shift the box 50 pixels to the right and 50 pixels up
            x = max(0, x - 50)
            y = max(0, y - 50)
            
            # # Adjust the box if it goes beyond the image dimensions
            # if x + w > image.shape[1]:
            #     w = image.shape[1] - x
            # if y + h > image.shape[0]:
            #     h = image.shape[0] - y

            # # Draw the bounding box and label on the image
            # label = f"{classes[class_id]}: {confidence:.2f}"
            # if label[0]=='n':
            #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # if label[0]=='s':
            #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 228, 0), 1)
             # Adjust the bounding box coordinates based on the face region
            if x + w > image.shape[1]:
                w = image.shape[1] - x
            if y + h > image.shape[0]:
                h = image.shape[0] - y

           
            # Calculate the adjusted top-left corner coordinates
            adjusted_x = max(0, x + (w // 2) - (image.shape[1] // 2))
            adjusted_y = max(0, y + (h // 2) - (image.shape[0] // 2))

            # Calculate the adjusted bounding box width and height
            adjusted_w = w
            adjusted_h = h

            
            
            # Draw the bounding box and label on the image
            label = f"{classes[class_id]}: {confidence:.2f}"
            if label[0] == 'n':
                cv2.rectangle(image, (adjusted_x, adjusted_y), (adjusted_x + adjusted_w, adjusted_y + adjusted_h), (0, 255, 0), 2)
            if label[0] == 's':
                cv2.rectangle(image, (adjusted_x, adjusted_y), (adjusted_x + adjusted_w, adjusted_y + adjusted_h), (0, 0, 255), 2)
            cv2.putText(image, label, (adjusted_x, adjusted_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 228, 0), 1)

# Display the output image
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
