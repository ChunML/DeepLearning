from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16
import argparse
import cv2
import numpy as np
import os
import random

import threading

label = ''
frame = None

class MyThread_2(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		global frame
		global label
		# Load the VGG16 network
		print("[INFO] loading network...")
		model = VGG16(weights="imagenet")

		if (~(frame is None)):
			while (~(frame is None)):
				image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
				image = image.transpose((2, 0, 1))
				image = image.reshape((1,) + image.shape)

				image = preprocess_input(image)
				preds = model.predict(image)
				(inID, label) = decode_predictions(preds)[0]


class MyThread_1(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		global frame
		global label

		cap = cv2.VideoCapture(0)
		if (cap.isOpened()):
			print("Camera OK")
		else:
			cap.open()

		while (True):
			ret, original = cap.read()

			frame = cv2.resize(original, (224, 224))

			# Display the predictions
			# print("ImageNet ID: {}, Label: {}".format(inID, label))
			cv2.putText(original, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
			cv2.imshow("Classification", original)

			if (cv2.waitKey(1) & 0xFF == ord('q')):
				break;

		cap.release()
		frame = None
		cv2.destroyAllWindows()

camera_thread = MyThread_1()
keras_thread = MyThread_2()

camera_thread.start()
keras_thread.start()