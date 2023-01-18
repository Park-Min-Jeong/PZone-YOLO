import os

import cv2
import numpy as np

# Load Yolo

def Detector(filename):
    BASE_DIR = "/home/YOLO-Test/FastModel/yolo"
    net = cv2.dnn.readNet(os.path.join(BASE_DIR,"backup/yolov4-tiny-custom_best.weights"),os.path.join(BASE_DIR,"backup/yolov4-tiny-custom.cfg"))
    with open(os.path.join(BASE_DIR,"backup/ClassNames.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    # colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading image
    # PHT_DIR = "C:/Users/smile/Documents/FastModel/photo"
    # img=Image.open(os.path.join(PHT_DIR,filename))
    # img.show()
    img = cv2.imread(filename)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > 0.5:
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(int(detection[0] * width) - w / 2)
                y = int(int(detection[1] * height) - h / 2)
                boxes.append([x, y, w, h])
                class_ids.append(class_id)

    return {"length": len(class_ids), "Kinds":list(set(class_ids))}
