import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO


def load_image(image_path):
    pillow_image = Image.open(image_path)
    numpy_image = np.array(pillow_image)
    cv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return cv_image


def draw_boxes(img, boxes):
    colors = np.random.uniform(0, 255, size=(3, 3))

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        class_id, class_name, x, y, w, h = boxes[i]
        color = colors[class_id]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, class_name, (x, y + 30), font, 3, color, 3)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_image(image_path):
    # load image
    img = load_image(image_path)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # get class name
    with open("backup/ClassNames.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    f.close()

    # load and run darknet
    net = cv2.dnn.readNet("backup/yolov4-tiny-custom_best.weights", "backup/yolov4-tiny-custom.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    net.setInput(blob)
    outs = net.forward(output_layers)

    # whether no parking zone
    class_ids = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > 0.5:
                class_ids.append(class_id)

                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(int(detection[0] * width) - w / 2)
                y = int(int(detection[1] * height) - h / 2)
                boxes.append([class_id, classes[class_id], x, y, w, h])

    draw_boxes(img, boxes)

    result = {
        "park": len(class_ids) < 0,  # False: cannot park, True: can park
        "class_id": list(set(class_ids)),
        "class_name": [classes[x] for x in set(class_ids)]
    }
    print(result)

    return result


# image_url = "https://cdn.pixabay.com/photo/2022/12/02/23/37/traffic-7631808_960_720.jpg"
# # image_url = "https://cdn.pixabay.com/photo/2021/11/13/19/46/elderly-man-6792215_960_720.jpg"
# res = requests.get(image_url)
# detect_image(BytesIO(res.content))

detect_image("backup/test_null.jpg")