import os

import cv2
from dotenv import load_dotenv

load_dotenv('../../.env')

from platform_sdk import SDK


if __name__ == "__main__":
    image = cv2.imread('../resource/face.jpg')

    model = SDK("ce7e5016-def7-439f-ba64-d5b4293553c7")
    model.set_params({'confidence_threshold': 0.7, 'iou_threshold': 0.3})
    pred = model(image)['detection']

    for coord in pred:
        image = cv2.rectangle(image, coord[:2], coord[2:4], (0, 0, 255), 2)

    if not os.path.exists('../results'):
        os.mkdir('../results')

    cv2.imwrite('../results/face_detection_result.jpg', image)
