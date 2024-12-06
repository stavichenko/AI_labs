import albumentations as A
import cv2

t = A.Compose([A.GaussNoise()])
for i in range(9):
    img = cv2.imread(f's{i+1}.png')
    img = t(image=img)['image']
    cv2.imshow('img', img)
    cv2.waitKey()

