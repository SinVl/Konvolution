import sys
import os
from os.path import dirname, abspath


import cv2
from dotenv import load_dotenv

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
load_dotenv('.env')

from konvolution import SDK


if __name__ == "__main__":
    image = cv2.imread('../resource/people.jpg')

    model = SDK("552d9c12-2640-4c07-842a-bd7e3980d715")
    model.set_params({'confidence_threshold': 0.6, 'iou_threshold': 0.3})
    pred = model(image)['boxes']
    print(pred)

    for coord in pred:
        image = cv2.rectangle(image, coord[:2], coord[2:4], (0, 0, 255), 2)

    if not os.path.exists('../results'):
        os.mkdir('../results')

    cv2.imwrite('../results/yolo_detection_result.jpg', image)
