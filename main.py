import cv2
import numpy as np
import pickle
from keras.models import load_model

####################
frameWidth = 100  # CAMERA RESOLUTION
frameHeight = 75
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Import the trained model
new_model = load_model('model_trained.h5')


def getCalssName(classNo):
    if classNo == 0:
        return 'Ceramic tiles non-crack'
    elif classNo == 1:
        return 'Ceramic tiles crack'


while True:
    # READ IMAGE
    _, imgOrignal = cap.read()
    if not _:
        continue
    imgOrignal = cv2.resize(imgOrignal, dsize=None, fx=0.5, fy=0.5)
    # PROCESS IMAGE
    img = imgOrignal.copy()
    img = cv2.resize(img, dsize=(400, 3))
    img = img.astype('float') * 1. / 255
    cv2.imshow("Processed Image", img)
    img = np.expand_dims(img, axis=0)
    cv2.putText(imgOrignal, "STATUS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = new_model.predict(img)
    classIndex = new_model.predict_classes(img)
    probabilityValue = new_model.amax(predictions)
    if probabilityValue > threshold:
        cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
