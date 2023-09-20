import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumourCategorical.h5')

image = cv2.imread('D:\Pranav\Brain Tumour Detection\pred\pred13.jpg')

img = Image.fromarray(image)
img = img.resize((64, 64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

print(img)

result = model.predict(input_img)
result_final = np.argmax(result, axis=1)


print(result_final)
