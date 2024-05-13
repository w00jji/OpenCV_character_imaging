# 19122039 김우진
import numpy as np
import cv2

image = cv2.imread('img/translate.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상 파일 읽기 에러")

h, w = image.shape

# flip1: 좌우로 뒤집기
flip1 = np.float32([[-1, 0, w],
                    [0, 1, 0]])

# flip2: 상하로 뒤집기
flip2 = np.float32([[1, 0, 0],
                    [0, -1, h]])

# flip3: 상하좌우 모두 뒤집기
flip3 = np.float32([[-1, 0, w],
                    [0, -1, h]])

dts1 = cv2.warpAffine(image, flip1, (w, h))
dst2 = cv2.warpAffine(image, flip2, (w, h))
dst3 = cv2.warpAffine(image, flip3, (w, h))

cv2.imshow('image', image)
cv2.imshow('dts1', dts1)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)
cv2.waitKey(0)
