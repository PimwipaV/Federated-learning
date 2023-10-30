import numpy as np
import cv2
import os
import argparse
from ultralytics import YOLO
from utils import load_annotations, calculate_ap

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="../yolov4-tiny.weights", help="Path to YOLOv4 weights file")
parser.add_argument("--anchors", default="path_to_anchors_file", help="Path to anchors file")
parser.add_argument("--classes", default="path_to_classes_file", help="Path to classes file")
parser.add_argument("--test-annotation", default="path_to_test_annotation_file", help="Path to test annotation file (YOLO format)")
parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold for detections")
parser.add_argument("--iou-threshold", type=float, default=0.4, help="IoU threshold for non-maximum suppression")
args = parser.parse_args()

# Initialize YOLO model
yolo = YOLO(model_path=args.model, anchors_path=args.anchors, classes_path=args.classes)

# Load test annotations
test_annotations = load_annotations(args.test_annotation)

# Initialize variables for evaluation
true_positives = 0
false_positives = 0
total_gt_objects = 0

# Loop through the test dataset
for image_path, gt_objects in test_annotations.items():
    # Load the image
    image = cv2.imread(image_path)

    # Get predictions from the YOLO model
    detections = yolo.detect_image(image, confidence_threshold=args.confidence_threshold)

    # Calculate true positives and false positives for this image
    true_pos, false_pos = calculate_ap(detections, gt_objects, args.iou_threshold)
    
    # Update counts
    true_positives += true_pos
    false_positives += false_pos
    total_gt_objects += len(gt_objects)

# Calculate precision and recall
precision = true_positives / (true_positives + false_positives)
recall = true_positives / total_gt_objects

# Calculate and display the Average Precision (AP)
ap = precision * recall
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Average Precision (AP): {ap:.4f}")

# Cleanup
yolo.close_session()
