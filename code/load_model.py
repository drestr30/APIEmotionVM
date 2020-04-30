from keras.models import load_model
from keras.backend import get_session, set_session
import tensorflow as tf
import cv2 as cv
import numpy as np

def preprocess_input(x, v2=True, f=False):
    x = x.astype('float32')

    if f:
        x = (255 - x) / 255.0
    else :
        x = x / 255.0

    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def init():
    sess = get_session()
    model = load_model('./code/models/emotion_model.hdf5')

    graph = tf.get_default_graph()

    return model, graph, sess


class FaceDetector:

    def __init__(self):
        self.prototxt = "./code/models/FaceDetection/deploy.prototxt.txt"
        self.model = "./code/models/FaceDetection/res10_300x300_ssd_iter_140000.caffemodel"
        self.net = cv.dnn.readNetFromCaffe(self.prototxt, self.model)

    def find_faces(self, image, confidence):
        (ih, iw) = image.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections and
        # predictions
        # print("[INFO] computing object detections...")
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = np.empty((np.sum(detections[0, 0, :, 2] > confidence), 4))
        for i in range(0, detections.shape[2]):
            dc = detections[0, 0, i, 2].max()

            #print("confidence: {:.2f}%".format(confidence * 100))

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if dc > confidence:
                 #raise ValueError("No valid faces detected")
            # compute the (x, y)-coordinates of the bounding box for the
            # object
                box = detections[0, 0, i, 3:7] * np.array([iw, ih, iw, ih])
                (startX, startY, endX, endY) = box.astype("int")
                face = (startX, startY, endX - startX, endY - startY)
                faces[i] = face

        return faces

