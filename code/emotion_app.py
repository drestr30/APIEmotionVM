from flask import Flask, request
import numpy as np
import sys
import os
import cv2 as cv
import tensorflow as tf
from load_model import *


# Initialize the app from Flask
app = Flask(__name__)

global model, graph

model, graph, sess = init()
detector = FaceDetector()
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def reshape_input(img):
    img = cv.resize(img,(48,48))
    img = img.reshape(1,48,48,1)

    return img

def build_response(face, output):
    emotions = np.squeeze(output, axis=0)
    emotions_dict = dict(zip(EMOTIONS, emotions.tolist()))
    face_message = ', '.join(str(p) for p in face)
    message = {'Face':face_message, 'Emotions':emotions_dict}
    return message

def build_error_response():
    message = {'Error': 'Not faces detected in input image.'}
    return message

# Define a route to hello_world function
@app.route('/hello')
def hello_world():
    return 'Hello World'

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = detector.find_faces(img, 0.7)
    if faces.size == 0:
        return build_error_response()

    if faces.shape[0] >= 1:
        for face in faces:
            (x, y, w, h) = face.astype(int)

            # Extract the ROI of the face from the grayscale image
            roi = gray[y:y + h, x:x + w]

            #resize it to the fixed model size, and then prepare
            # the ROI for classification via the CNN

            roi = reshape_input(roi)
            print(roi.max())
            imgData = preprocess_input(roi)
            print(imgData.max())
            with sess.as_default():
                with graph.as_default():
                    out = model.predict(imgData)
                    response = build_response(face, out)
                    return response

if __name__ == '__main__':
    port = 8888
    app.run(host='0.0.0.0', port=port)
