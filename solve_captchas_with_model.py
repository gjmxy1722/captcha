from keras.models import load_model
from imutils import paths
import numpy as np
import os
import imutils
import cv2
import pickle
import cut_image_code as dlp
from PIL import Image


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "images"


# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))

# loop over the image paths
for image_file in captcha_image_files:
    # Load the image and convert it to grayscale
    train=0
    image = Image.open(image_file)  # 打开文件
    cut_image_dir = r"./cut_images/"
    filename = r"C:/Users/jwt/Desktop/captcha_code//0123.png"
    dlp.two_value(image, filename,cut_image_dir,train)


    letters = []
    for i in range(4):
        letter_image = cv2.imread('./cut_images/'+str(i)+'.png')
        letter_image = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)

        # Add a third channel dimension to the image to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)
        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0][0]
        letters.append(letter)

    captcha_text = "".join(letters)
    print("图片名称为：",image_file.split('_')[2],"CAPTCHA text is: {}".format(captcha_text))

