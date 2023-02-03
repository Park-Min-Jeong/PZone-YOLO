import os
import cv2
import numpy as np

# Load Yolo

def Detector(filename, modelname):
    # BASE_DIR = "/home/YOLO-Test/yolo"
    BASE_DIR = "d:/study/Git/YOLO-Test/yolo"
    net = cv2.dnn.readNetFromDarknet(os.path.join(BASE_DIR, f"backup/{modelname}.cfg"), os.path.join(BASE_DIR, f"backup/{modelname}_best.weights"))
    with open(os.path.join(BASE_DIR, f"backup/{modelname}.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

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
    class_names = []
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
                class_names.append(classes[int(class_id)])
    # for i in range(len(boxes)):
    #     box = boxes[i]
    #     cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color=colors[class_ids[i]], thickness=2)
    #
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    return {"length": len(class_ids), "Kinds": list(set(class_ids))}


def Score(filename):
    s_out = Detector(filename, "surface")
    k_out = Detector(filename, "kickboard")

    if k_out["length"] == 0:
        s_out["Kickboard"] = 0
    else:
        s_out["Kickboard"] = 1

    return s_out