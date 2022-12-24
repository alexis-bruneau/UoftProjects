# USAGE
# python part2_yolo.py --image images/baggage_claim.jpg --yolo yolo

# import the necessary packages
import numpy as np
import pandas
import time
import cv2
import os
from numpy import array, average

###########################################################
# OPTIONS
###########################################################
image_path = 'data/train/left/000001.png'
yolo_dir = 'yolo'

# minimum probability to filter weak detections
confidence_th = 0.5 

# threshold when applyong non-maxima suppression
threshold = 0.5
###########################################################
def detect_object(file_path,confidence_th,threshold,show_result=True):
    image_path = file_path
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([yolo_dir, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                            dtype="uint8")

    # derive the paths to the YOLO weights and model configurationY
    weightsPath = os.path.sep.join([yolo_dir, "yolov3.weights"])
    configPath = os.path.sep.join([yolo_dir, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    #ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_th:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_th,
                            threshold)

    box_car = []
    car_confidence = [] 
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates only for car
            if LABELS[classIDs[i]] == 'car':
                box_car.append(boxes[i])
                car_confidence.append(confidences[i])
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
                    
    # show the output image
    if show_result == True:
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    

    box_car = np.array(box_car)

    return image,box_car, car_confidence


    

def detect_train_test(Train=True,show_result=True):

    if Train == True:
        # Test values to find the best one 
        #confidence_th_list = [0.1, 0.25, 0.5, 0.75, 0.9]
        #threshold_list = [0.1, 0.25, 0.5, 0.75 , 0.9]
        confidence_th_list = [0.5]
        threshold_list = [0.5]
        accuracy_dir = 'data/train/Best_accuracy/Accuracy'
        accuracy_dir = accuracy_dir + ".txt"
        accuracy_dir = open(accuracy_dir, "a")    
        accuracy_dir.truncate(0)
    else:
        # Based on test
        confidence_th_list = [0.5]
        threshold_list = [0.5]


    for confidence_th in confidence_th_list:
        for threshold in threshold_list:


            if Train == True:
                image_path = 'data/train/left'
                output_dir = 'data/train/estimated_box'
                ground_truth_dir = 'data/train/gt_labels'                
                sample_list = ['000001', '000002', '000003', '000004', '000005','000006','000007','000008','000009','000010']
            
            else: 
                image_path = './data/test/left'
                output_dir = './data/test/estimated_box'
                ground_truth_dir = 'data/test/gt_labels'
                sample_list = ['000011', '000012', '000013', '000014', '000015']

            # Run yolo_box for each images
            car_accuracy = [] 
            miss_box = 0 # Count number of missed labelled cars            
            

            for sample in sample_list:
                file_path = image_path +  "/" + sample +'.png'
                print(file_path)
                image, box, car_confidence = detect_object(file_path,confidence_th,threshold,show_result)
                for result in car_confidence:
                    car_accuracy.append(result)
                #car_accuracy.append(car_confidence)        
                # Saving results
                output_path = output_dir + "/" + sample +'.png'
                output_path_box = open(output_dir + "/" + sample + '.txt', "a")
                output_path_box.truncate(0)
                cv2.imwrite(output_path, image)

                # Return x,y,w,h values
                for box_dim in box:
                    line = "{:.2f} {:.2f} {:.2f} {:.2f}".format(box_dim[0], box_dim[1], box_dim[2], box_dim[3])
                    output_path_box.write(line + '\n')
            
                # Calculate accuracy
                if Train == True:
                    gt_file_path = ground_truth_dir + "/" + sample + '.txt'
                    gt_file = open(gt_file_path, 'r')
                    Lines = gt_file.readlines()
                    count = 0
                    for line in Lines:
                        # Get x and y values of gt car boxes
                        if line.split()[0] == "Car":
                            try:
                                x = (box[count][0])
                                y = (box[count][1])
                            except:
                                miss_box += 1 # if the index is out of range, it means that the car was not caputres
                            count += 1
            
            if Train == True:
                car_accuracy = array(car_accuracy)
                car_average_accuracy = np.mean(car_accuracy)
                print(f"The average accuracy was {car_average_accuracy} and {miss_box} bounding box was missed for threshold = {threshold} and confidence threshold = {confidence_th}")           
                accuracy_dir.write(f"The average accuracy was {car_average_accuracy} and {miss_box} bounding box was missed for threshold = {threshold} and confidence threshold = {confidence_th}" + '\n')
        


if __name__ == "__main__":
    # Train set to True to see results of training set, show results to see bounding box
    detect_train_test(Train=True,show_result=True) 