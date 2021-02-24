import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
import time

parser = argparse.ArgumentParser()
parser.add_argument('--video', help='Path of video.',
                    default='./data/clips/test_clip.h264')

args = parser.parse_args()


MODEL_NAME = './Lime_TFLite_model'
VIDEO_PATH = args.video
min_conf_threshold = 0.8

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Path to .tflite file, which contains the model that is used for object detection
tflite_list = ['detect.tflite', 'detect_dynamic_range.tflite', 'detect_float16.tflite']
PATH_TO_CKPT = os.path.join(MODEL_NAME, tflite_list[0])

# Path to label map file
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'labelname.txt')

# Load the label map
labels = []
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

isEntry = False

marker_radius_sum = 0
marker_radius_avg = 0

lime_count = 0
marker_count = 0

limes = []

sqsize = 320
imH, imW = sqsize, sqsize
margin = 45
ptx = int(sqsize * 60/100)
pt1 = (( ptx - margin, 0 ), ( ptx - margin, int(sqsize) ))
pt2 = (( ptx + margin, 0 ), ( ptx + margin, int(sqsize) ))

def overlay_image(image, classes, boxes, scores):

    # red lines
    cv2.line(image, pt1[0], pt1[1], (0, 0, 255), 1)
    cv2.line(image, pt2[0], pt2[1], (0, 0, 255), 1)        

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text


    # for box in confirmed_boxes:
    #     xmin = box[1]
    #     xmax = box[3]
    #     radius = (box[3] - box[1]) * 320

    #     if isEntry is False and int(xmin * 320) > int(sqsize/2-margin) and int(xmin * 320):
    #         isEntry = True
                    
    #     if isEntry and int(xmax * 320) > int(sqsize/2-margin) and int(xmax * 320) < int(sqsize/2+margin):
    #         isEntry = False
    #         print("Current Time:",time.asctime(time.localtime(time.time())))
    #         if classes[0] == 0:
    #             lime_count = lime_count + 1
    #             lime_size = (radius / marker_radius_avg) * 40
    #             lime_size = round(lime_size, 2)
    #             limes.append(lime_size)
    #             print("lime count:", lime_count, ", size:", lime_size, "mm.")
    #         if classes[0] == 1:
    #             marker_count = marker_count + 1
    #             marker_radius_sum = marker_radius_sum + radius
    #             marker_radius_avg = marker_radius_sum / marker_count
    #             print("marker count:", marker_count)
                    
    #         if int(xmax * 320) < int(sqsize/2-margin): # reset counter
    #             isEntry = False 


def detect(image):
    input_data = np.expand_dims(image, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    print("Elapsed time:", elapsed_time, "seconds")

    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    return classes[scores > min_conf_threshold], boxes[scores > min_conf_threshold], scores[scores > min_conf_threshold]

def count_object()

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    default_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    default_height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )

    new_width = sqsize
    new_height = int(default_height * (new_width / default_width))
    count = 0
    while True:
        ret, raw_img = cap.read()
        if not ret:
            break

        raw_img = cv2.resize(raw_img, (new_width, new_height))
        frame = np.zeros((sqsize, sqsize, 3), np.uint8)
        if new_width > new_height:
            offset = int( (new_width - new_height) /2 )
            frame[offset:new_height+offset,:] = raw_img
        else:
            offset = int( (new_height - new_width) /2 )
            frame[:,offset:] = raw_img
        
        if count == 5:
            # Detecting objects
            classes, boxes, scores = detect(frame)
            overlay_image(frame, classes, boxes, scores)
            count = 0
            cv2.imshow('Preview', frame)

        count = count + 1
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == 32: # For debugging
            print("***************************")
    
    cap.release()


main()