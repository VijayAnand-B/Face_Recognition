# ------------------------- FACE-ATTENDENCE SYSTEM ---------------------------#

# Import Libraries
from cv2 import cv2
import time
import gspread
import numpy as np
import face_recognition
import os
from datetime import datetime
import requests

# Update Attedance at
def markAttendance(name):
    # col = sheet.col_values(1)  # data in Column 1
    # i = len(col)
    # i += 1
    now = datetime.now()  # get present time
    time = now.strftime("%H:%M:%S")  # Time format
    # if name not in col:
        # sheet.update(f'A{i}', time)  # update Name at 'A' column 'i'th row
        # sheet.update(f'B{i}', name)

    URL1 = "https://script.google.com/macros/s/AKfycbx9uwImueyD_LD2HGSvj9-5jqjuymW9AFGF6fzmPAkM0noVVK4/exec"
    parameters = {'sheetname': 'Face', 'value': name}
    response1 = requests.get(URL1, params=parameters)
    print(response1)

# Read images in 'images' folder
path = 'Database'
images = []  # LIST CONTAINING ALL THE IMAGES
className = []  # LIST CONTAINING ALL THE CORRESPONDING CLASS Names
myList = os.listdir(path)
print("Total Students of Class A:", len(myList))
for imgDir in myList:
    curImg = cv2.imread('{}/{}'.format(path,imgDir))
    images.append(curImg)
    className.append(os.path.splitext(imgDir)[0])


# To find Face Encodings (128 features extraction from face)
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)

# Live Webcam
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    # Resize img smaller for better accuracy
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    # RGB format is needed for processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        # Known Face is detected
        if faceDis[matchIndex] < 0.50:
            name = className[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Resize back to original size
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 3, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            time.sleep(10)
            markAttendance(name)

        else:  # If unknown Face is detected
            name = 'Unknown'
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 3, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    cv2.imshow('face', img)
    if cv2.waitKey(1) == ord('q'):  # Exit by pressing 'q'
        break
