import cv2
from dotenv import load_dotenv

load_dotenv('../../.env')

from platform_sdk import SDK


if __name__ == "__main__":
    model = SDK("ce7e5016-def7-439f-ba64-d5b4293553c7")
    model.set_params({'confidence_threshold': 0.7, 'iou_threshold': 0.3})

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        pred = model(frame)

        dets = pred['detection']
        if len(dets) != 0:
            dets = dets[0]
        else:
            continue
        face_img = frame[dets[1]:dets[3], dets[0]:dets[2]]

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('frame', face_img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
