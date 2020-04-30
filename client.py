import requests
import json
import cv2
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

addr = 'http://localhost:8888'
test_url = addr + '/predict'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}


#starting video streaming
print("[INFO] starting video stream...")
video_capture = VideoStream(src=0).start()
time.sleep(5.0)
frame_size = np.shape(video_capture.read())

## Send one image to the service
# one_img = cv2.imread(IMG_PATH)
# _, img_encoded = cv2.imencode('.jpg', one_img)
## send http request with image and receive response
# response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
## decode response
# print(json.loads(response.text))

# for video webcam
fig = plt.figure()
fig.show()
while True:
    frame = video_capture.read()

    frame = imutils.resize(frame, height=500)
    height, width, _ = np.shape(frame)

    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', frame)
    # send http request with image and receive response
    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)

    # decode response
    response_json = json.loads(response.text)
    try:
        emotions, face = response_json.values()
        face = np.asarray(face.split(', '), dtype=float)

        # Draw rectangle around face
        (x, y, w, h) = face.astype(int)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # emotional barplot
        fig.clf()
        plt.bar(list(emotions.keys()), list(emotions.values()))
        plt.pause(0.05)
        fig.canvas.draw()

    except ValueError:
        error = response_json['Error']
        print(error)
        pass

    cv2.imshow('procesed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # expected output: {u'message': u'image received. size=124x124'}