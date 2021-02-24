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
min_conf_threshold = 0.7

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Path to .tflite file, which contains the model that is used for object detection
tflite_list = ['detect.tflite', 'detect_dynamic_range.tflite', 'detect_float16.tflite']
PATH_TO_CKPT = os.path.join(MODEL_NAME, tflite_list[0])

# Path to label map file
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'labelmap.txt')

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

marker_diameter_sum = 0
marker_diameter_avg = 0

lime_count = 0
marker_count = 0

limes = []
time_used_per_objects = []

sqsize = 320
imH, imW = sqsize, sqsize
margin = 40
ptx = int(sqsize * 50/100)
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

def detect(image):
    input_data = np.expand_dims(image, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    # print("Elapsed time:", elapsed_time, "seconds")

    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    return classes[scores > min_conf_threshold], boxes[scores > min_conf_threshold], scores[scores > min_conf_threshold]

def main():
    global marker_diameter_sum
    global marker_diameter_avg
    global lime_count
    global marker_count
    global lime
    global time_used_per_objects

    start_exec_time = time.time()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    default_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    default_height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )

    new_width = sqsize
    new_height = int(default_height * (new_width / default_width))
    xleft, xright = pt1[0][0], pt2[0][0]
    isEntry = False
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
            start = time.time()
            classes, boxes, scores = detect(frame)
            end = time.time()
            elapsed_time = end - start
            overlay_image(frame, classes, boxes, scores)
            
            for box in boxes:
                xmin = int(box[1] * sqsize)
                xmax = int(box[3] * sqsize)
                ymin = int(box[0] * sqsize)
                ymax = int(box[2] * sqsize)
                diameter = (xmax - xmin) + (ymax - ymin) / 2

                if isEntry is False and xmin > xleft  and xmin < xright:
                    isEntry = True
                            
                if isEntry and xmin < xleft and xmax > xleft and xmax < xright:
                    isEntry = False
                    if classes[0] == 0:
                        lime_count = lime_count + 1
                        lime_size = (diameter / marker_diameter_avg) * 40
                        lime_size = round(lime_size, 2)
                        limes.append(lime_size)
                        print("lime count:", lime_count, ", diameter size:", lime_size, "mm.")
                        time_used_per_objects.append(elapsed_time)
                    if classes[0] == 1:
                        marker_count = marker_count + 1
                        marker_diameter_sum = marker_diameter_sum + diameter
                        marker_diameter_avg = marker_diameter_sum / marker_count
                        print("marker count:", marker_count)
                        time_used_per_objects.append(elapsed_time)

                if isEntry and xmax < xleft: # reset counter
                    isEntry = False 

            count = 0
            cv2.imshow('Preview', frame)

        count = count + 1
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == 32: # For debugging
            print("***************************")
            print("marker diameter avg", marker_diameter_avg)
    
    cap.release()
    end_exec_time = time.time()
    exec_time = round(end_exec_time - start_exec_time, 2)
    avg_time_per_object = round(sum(time_used_per_objects) / len(time_used_per_objects), 2)
    avg_lime_size = round(sum(limes) / len(limes), 2)
    print("*************************")
    print("Execution time:", exec_time, "seconds")
    print("Lime counts:", lime_count)
    print("Average lime size (diameter):", avg_lime_size,"mm")
    print("Average execution time per object:", avg_time_per_object, "seconds")
    print("*************************")



main()