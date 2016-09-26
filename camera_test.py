from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16
import argparse
import cv2
import numpy as np
import os
import random
import sys

import threading

label = ''
frame = None

class MyThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		global label
		# Load the VGG16 network
		print("[INFO] loading network...")
		self.model = VGG16(weights="imagenet")

		while (~(frame is None)):
			(inID, label) = self.predict(frame)

	def predict(self, frame):
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
		image = image.transpose((2, 0, 1))
		image = image.reshape((1,) + image.shape)

		image = preprocess_input(image)
		preds = self.model.predict(image)
		return decode_predictions(preds)[0]

cap = cv2.VideoCapture(0)
if (cap.isOpened()):
	print("Camera OK")
else:
	cap.open()

keras_thread = MyThread()
keras_thread.start()

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
sys.exit()