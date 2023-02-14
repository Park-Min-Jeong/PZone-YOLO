import os
import cv2
import numpy as np
import math
import base64


def imageEncode(filename):
    image = cv2.imread(filename)
    is_success, img_buf_arr = cv2.imencode(".jpg", image)
    encoded_img = base64.b64encode(img_buf_arr.tobytes())  # base64로 변환
    uri = f"data:image/*;base64,{str(encoded_img)[2:-2]}"

    return uri


def showBoxImage(image, result):
    height, width, channels = image.shape
    colors = np.random.uniform(0, 255, size=(len(result.keys()), 3))
    show_img = image.copy()
    for idx, box_list in enumerate(result.values()):
        for box in box_list:
            x, y, w, h = [int(box[i]*width) if i%2==0 else int(box[i]*height) for i in range(0, 4)]
            cv2.rectangle(show_img, (x, y), (x+w, y+h), color=colors[idx], thickness=2)
    cv2.imshow("image", show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Detector(filename, modelname):
    # Loading model
    BASE_DIR = "/home/YOLO-Test/yolo"
    # BASE_DIR = "d:/study/Git/YOLO-Test/yolo"
    net = cv2.dnn.readNetFromDarknet(os.path.join(BASE_DIR, f"backup/{modelname}.cfg"), os.path.join(BASE_DIR, f"backup/{modelname}_best.weights"))
    with open(os.path.join(BASE_DIR, f"backup/{modelname}.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Loading image
    img = cv2.imread(filename)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    yolo_result = {className: [] for className in classes}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > 0.5:
                yolo_result[classes[int(class_id)]].append(
                    list(np.clip([
                        detection[0] - detection[2] / 2,  # min_x(left)
                        detection[1] - detection[3] / 2,  # min_y(top)
                        detection[2],  # width
                        detection[3],  # height
                        detection[0],  # center_x
                        detection[1]  # center_y
                    ], 0, 1))
                )

    # boxed image
    # showBoxImage(img, yolo_result)

    return yolo_result


def Score(filename):
    surface_result = Detector(filename, "surface")
    kickboard_result = Detector(filename, "kickboard")

    image_distance = {"sidewalk": 0, "crosswalk": 0, "braille_block": 0, "bike_lane": 0}
    wheel_list = []

    # 킥보드가 없는 경우
    if len(kickboard_result["kickboard"]) == 0:
        if len(kickboard_result["wheel"]) == 0:
            kickboard = False
        else:
            kickboard = True
            wheel_array = np.array(kickboard_result["wheel"])
            max_w_idx, max_h_idx = np.argmax(wheel_array, axis=0)[2:4]
            idx = max_w_idx
            if max_w_idx != max_h_idx:
                max_w, max_h = wheel_array[max_w_idx, 2], wheel_array[max_h_idx, 3]
                avg_w, avg_h = np.average(np.array(kickboard_result["wheel"]), axis=0)[2:4]
                if max_w - avg_w < max_h - avg_h:
                    idx = max_h_idx

            w_box = wheel_array[idx]
            wheel_list.append(w_box)

    # 킥보드가 있는 경우
    else:
        kickboard = True
        # 박스의 길이와 너비가 가장 큰 킥보드
        kickboard_array = np.array(kickboard_result["kickboard"])
        max_w_idx, max_h_idx = np.argmax(kickboard_array, axis=0)[2:4]
        idx = max_w_idx
        if max_w_idx != max_h_idx:
            max_w, max_h = kickboard_array[max_w_idx, 2], kickboard_array[max_h_idx, 3]
            avg_w, avg_h = np.average(np.array(kickboard_result["kickboard"]), axis=0)[2:4]
            if max_w - avg_w < max_h - avg_h:
                idx = max_h_idx
        k_box = kickboard_array[idx]

        # 킥보드와 겹쳐지는 바퀴
        for box in kickboard_result["wheel"]:
            if (box[0] > k_box[0] + k_box[2]) or (box[1] > k_box[1] + k_box[3]) or (box[0] + box[2] < k_box[0]) or (box[1] + box[3] < k_box[1]):
                continue
            wheel_list.append(box)


    # 바퀴와 박스 간 최소 거리 계산 -> 1 - 최소 거리로 점수 계산
    # score = 0
    if kickboard == True:
        for key in surface_result.keys():
            min_distance = 1.5
            for w_box in wheel_list:  # 바퀴 여러 개
                for box in surface_result[key]:  # 금지구역 마커
                    if key == "sidewalk":  # 보도블럭은 바퀴가 보도블럭 위에 있을 때만 거리 계산
                        if (box[0] > w_box[0] + w_box[2]) or (box[1] > w_box[1] + w_box[3]) or (box[0] + box[2] < w_box[0]) or (box[1] + box[3] < w_box[1]):
                            continue
                    distance = math.dist((w_box[4], box[4]), (w_box[5], box[5]))  # 유클리드 거리
                    if distance < min_distance:
                        min_distance = distance
                        image_distance[key] = min_distance


    """
        if len(kickboard_result["kickboard"]) == 0:

        if len(kickboard_result["wheel"]) == 0:
            kickboard = False
        else:
            kickboard = True
            wheel_array = np.array(kickboard_result["wheel"])
            max_w_idx, max_h_idx = np.argmax(wheel_array, axis=0)[2:4]
            idx = max_w_idx
            if max_w_idx != max_h_idx:
                max_w, max_h = wheel_array[max_w_idx, 2], wheel_array[max_h_idx, 3]
                avg_w, avg_h = np.average(np.array(kickboard_result["wheel"]), axis=0)[2:4]
                if max_w - avg_w < max_h - avg_h:
                    idx = max_h_idx

            w_box = wheel_array[idx]
            wheel_list.append(w_box)

    # 킥보드가 있는 경우
    else:
        kickboard = True
        # 박스의 길이와 너비가 가장 큰 킥보드
        kickboard_array = np.array(kickboard_result["kickboard"])
        max_w_idx, max_h_idx = np.argmax(kickboard_array, axis=0)[2:4]
        idx = max_w_idx
        if max_w_idx != max_h_idx:
            max_w, max_h = kickboard_array[max_w_idx, 2], kickboard_array[max_h_idx, 3]
            avg_w, avg_h = np.average(np.array(kickboard_result["kickboard"]), axis=0)[2:4]
            if max_w - avg_w < max_h - avg_h:
                idx = max_h_idx
        k_box = kickboard_array[idx]

        # 킥보드와 겹쳐지는 바퀴
        for box in kickboard_result["wheel"]:
            if (box[0] > k_box[0] + k_box[2]) or (box[1] > k_box[1] + k_box[3]) or (box[0] + box[2] < k_box[0]) or (box[1] + box[3] < k_box[1]):
                continue
            wheel_list.append(box)


    # 바퀴와 박스 간 최소 거리 계산 -> 1 - 최소 거리로 점수 계산
    # score = 0
    if kickboard == True:
        for key in surface_result.keys():
            min_distance = 1.5
            for w_box in wheel_list:  # 바퀴 여러 개
                for box in surface_result[key]:  # 금지구역 마커
                    if key == "sidewalk":  # 보도블럭은 바퀴가 보도블럭 위에 있을 때만 거리 계산
                        if (box[0] > w_box[0] + w_box[2]) or (box[1] > w_box[1] + w_box[3]) or (box[0] + box[2] < w_box[0]) or (box[1] + box[3] < w_box[1]):
                            continue
                    distance = math.dist((w_box[4], box[4]), (w_box[5], box[5]))  # 유클리드 거리
                    if distance < min_distance:
                        min_distance = distance
                        

            if min_distance > 1:
                image_score[key] = 0
            else:
                image_score[key] = 1 - min_distance
    print(os.path.getsize(filename))
    """

    return {"kickboard": kickboard,
            "image_distance": image_distance,
            "uri": imageEncode(filename)}