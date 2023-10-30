!pip install unrar

from google.colab import drive
drive.mount('/content/gdrive')

!unrar x /content/gdrive/MyDrive/'Stroke Detection.rar'



cd content/Stroke\ Detection/YOLOv4-Tiny-Training/darknet

change in Makefile 
GPU=0
CUDNN=0
CUDNN_HALF=0

!make

then ./darknet works but complained that cannot find data/coco.names

in darknet/cfg/coco.data, change names = ../data/obj.names

  29 conv     21       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x  21 0.004 BF
  30 yolo
[yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
nms_kind: greedynms (1), beta = 0.600000 
  31 route  27 		                           ->   13 x  13 x 256 
  32 conv    128       1 x 1/ 1     13 x  13 x 256 ->   13 x  13 x 128 0.011 BF
  33 upsample                 2x    13 x  13 x 128 ->   26 x  26 x 128
  34 route  33 23 	                           ->   26 x  26 x 384 
  35 conv    256       3 x 3/ 1     26 x  26 x 384 ->   26 x  26 x 256 1.196 BF
  36 conv     21       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x  21 0.007 BF
  37 yolo
[yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
nms_kind: greedynms (1), beta = 0.600000 
Total BFLOPS 6.789 
avg_outputs = 299797 
Loading weights from ../yolov4-tiny.weights...
 seen 64, trained: 0 K-images (0 Kilo-batches_64) 
Done! Loaded 38 layers from weights-file 
 Detection layer: 30 - type = 28 
 Detection layer: 37 - type = 28 
../../../acute_ischaemic_stroke.jpg: Predicted in 1031.822000 milli-seconds.
OpenCV exception: show_image_cv 

Colab paid products - Cancel contracts here

import glob

myPath = '../data/Dataset/'
jpgCounter = len(glob.glob1(myPath,"*.jpg"))
jpgCounter
3749

มี training data 3749 files


#convert dataset prepared for YOLOv4-tiny into keras dataset format

import os

dataset_dir = 'path/to/dataset'
keras_labels_dir = 'path/to/keras_labels'

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


