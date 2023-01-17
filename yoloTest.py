def test_return_result():
    import json

    with open("results/output.json", "r") as file:
        result = json.load(file)
    file.close()

    if len(result[0]["objects"]) == 0:
        print("You can park here")
        return 1

    else:
        print("You cannot park here")
        for obj in result[0]["objects"]:
            print(obj["class_id"], end=" ")
        return 0


def test():
    import subprocess
    import shlex
    import os
    result_file = "./results/output.json"
    data_file = "./data/maskDatas.data"
    image_file = "./data/test.jpg"
    model_file = "./backup/yolov4-tiny-custom.cfg"
    weight_file = "./backup/yolov4-tiny-custom_final.weights"
    command = f'time ./darknet.exe detector test {data_file} {model_file} {weight_file} {image_file} -ext_output -dont_show -save_labels -out {result_file}'
    temp = subprocess.run(command,
                          shell=True,
                          check=True)
    print(temp.returncode)
    # test_return_result()


def test_null():
    import subprocess
    import os
    subprocess.run("time ./darknet detector test data/maskDatas.data backup/yolov4-tiny-custom.cfg backup/yolov4-tiny-custom_final.weights -ext_output -dont_show -save_labels -out results/output.json data/test_null.jpg", check=True)
    test_return_result()


test()
# test_null()