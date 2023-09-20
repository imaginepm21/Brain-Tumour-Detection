import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

image_dir = 'datasets/'

no_tumour = os.listdir(image_dir + 'no/')
yes_tumour = os.listdir(image_dir + 'yes/')

dataset = []
label = []

# print(no_tumour)

for i, image_name in enumerate(no_tumour):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_dir + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumour):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_dir + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        label.append(1)

print(dataset)
# print(label)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(
    dataset, label, test_size=0.2, random_state=0)

# x_train.shape, y_train.shape

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

Y_train = to_categorical(y_train, num_classes=2)
Y_test = to_categorical(y_test, num_classes=2)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# 1 because out output is binary classification, 2 if categorical classification
model.add(Dense(1))  
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, verbose=True, epochs=10,
          validation_data=(x_test, y_test), shuffle=False)

model.save('BrainTumourBinary.h5')
