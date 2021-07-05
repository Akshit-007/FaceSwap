import cv2
from PIL import Image
import numpy as np

harcascade_path = "../Thug_Life_filter/haarcascade_frontalface_default.xml"
modi_path = "mdi2_adobespark.png"

detector = cv2.CascadeClassifier(harcascade_path)

modi = Image.open(modi_path)

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()
    cut = frame
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray)
        background = Image.fromarray(frame)

        for(x, y, w, h) in faces:

            resized_mask = modi.resize((w, h), Image.ANTIALIAS)

            offset = (x, y)
            background.paste(resized_mask, offset, mask=resized_mask)

            frame = np.asarray(background)

        cv2.imshow("Swap_face", frame)

    key = cv2.waitKey(1)

    if(key == ord("q")):
        break


cap.release()
cv2.destroyAllWindows()
