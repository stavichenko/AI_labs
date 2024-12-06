import cv2
import tensorflow as tf
from keras import layers, models, optimizers
import albumentations as A
import numpy as np
import cv2
import random

s = 3
np.random.seed(s)
random.seed(s)

train_data = []

trans = A.Compose([
    A.GaussNoise(p=1),
    # A.GridDistortion(),
    # A.OpticalDistortion(),
])

CLASSES = ['S', 'G']

def make_dataset(n):
    data = []
    labels = []
    for i in range(n):
        img = np.zeros([25, 25]).astype('uint8')
        dx = np.random.randint(0, 5)
        dy = 25 - np.random.randint(0, 5)
        size = np.random.rand() + 1
        img = cv2.putText(img, 'G' if i % 2 else 'S', (dx, dy), cv2.FONT_HERSHEY_PLAIN, size, 255)
        img = trans(image=img)['image']
        data.append(img)
        l = np.zeros([2])
        l[i % 2] = 1
        labels.append(l)

    return np.array(data).reshape([n, 25, 25, 1]), np.array(labels)


def load_handcrafted(augment=True):
    n = 9
    data = []
    labels = []
    for i in range(n):
        lab = np.zeros([2])
        l = 0
        img = cv2.imread(f's{i + 1}.png')
        if img is None:
            img = cv2.imread(f'g{i + 1}.png')
            l = 1
            
        lab[l] = 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if augment:
            img = trans(image=img)['image']
        data.append(img)
        labels.append(lab)

    return np.array(data).reshape([n, 25, 25, 1]), np.array(labels)


def show(data, n=10):
    for i in range(min(n, len(data))):
        img = data[i, :, :, 0]
        cv2.imshow('img', img)
        cv2.waitKey()


train_x, train_y = make_dataset(300)
val_x, val_y = make_dataset(20)
test_x, test_y = load_handcrafted()

print('train')
# show(train_x)

model = models.Sequential([
    layers.Input([25, 25, 1]),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizers.Adam(1e-3))
model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), epochs=10)

for i in range(len(test_x)):
    img = test_x[i:i+1]
    lab = model.predict(img)[0]
    lab_true = test_y[i]
    correct = np.argmax(lab) == np.argmax(lab_true)
    img = img[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.copyMakeBorder(img, 3, 3, 3, 3, borderType=cv2.BORDER_CONSTANT, value=[0, 255, 0] if correct else [0, 0, 255])
    cv2.imshow(f'{i+1}   {CLASSES[np.argmax(lab)]}', img)
cv2.waitKey()