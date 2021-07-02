import cv2
import glob


# image detections and resizing
#
# images = glob.glob("*.jpg")
#
# for image in images:
#     img = cv2.imread(image, 0)
#     resized = cv2.resize(img, (100, 100))
#     cv2.imshow("Hey", resized)
#     cv2.waitKey(500)
#     cv2.destroyAllWindows()
#     cv2.imwrite("resized_"+image, resized)

# #########################********#####################

# Face detection
#
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#
# img = cv2.imread("news.jpg",1)
# gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# faces = face_cascade.detectMultiScale2(img, scaleFactor=1.1, minNeighbors=5)
#
# for x, y, w, h in faces[0]:
#     image = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
# print(faces)
# resized_img = cv2.resize(img,(int(gray_image.shape[1]/2),int(gray_image.shape[0]/2)))
# cv2.imshow("shivaji", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ###################***********######################

import time, pandas
from datetime import datetime

first_frame = None
status_list=[None,None]
times=[]
df = pandas.DataFrame(columns=["start","end"])
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)
while True:
    status=0

    check, frame = video.read()
    faces = face_cascade.detectMultiScale2(frame, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces[0]:
        image = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (21, 21), 0)
    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame=cv2.threshold(delta_frame,21,255,cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame,None,iterations=2)

    (cnt,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnt:
        if cv2.contourArea(contour) < 10000:

            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

        status_list=status_list[-2:]

    if status_list[-1] == 1 and status_list[-2] == 0 or status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    status_list.append(status)

    cv2.imshow("video", gray)
    cv2.imshow("delta", delta_frame)
    cv2.imshow("thresh", thresh_frame)
    cv2.imshow("video1",frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        if status == 1:
            times.append(datetime.now())

        break
print(status_list)
print(times)

for i in range(0,len(times), 2):
    df = df.append({"start":times[i],"end":times[i+1]},ignore_index=True)

df.to_csv("time.csv")
video.release()
cv2.destroyAllWindows()