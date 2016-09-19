from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16
import argparse
import cv2
import numpy as np
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True)
args = vars(ap.parse_args())

files = [os.path.join(args["folder"], f) for f in os.listdir(args["folder"])]
random.shuffle(files)


# Load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights="imagenet")

for file in files:
	# Load the image using OpenCV
	orig = cv2.imread(file)

	# Load the image using Keras helper ultility
	print("[INFO] loading and preprocessing image...")
	image = image_utils.load_img(file, target_size=(224, 224))
	image = image_utils.img_to_array(image)

	# Convert (3, 224, 224) to (1, 3, 224, 224)
	# Here "1" is the number of images passed to network
	# We need it for passing batch containing serveral images in real project
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)


	# Classify the image
	print("[INFO] classifying image...")
	preds = model.predict(image)
	(inID, label) = decode_predictions(preds)[0]

	# Display the predictions
	print("ImageNet ID: {}, Label: {}".format(inID, label))
	cv2.putText(orig, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	cv2.imshow("Classification", orig)
	cv2.waitKey(0)