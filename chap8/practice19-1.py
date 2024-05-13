import numpy as np
import cv2

def draw_bar(img, pt, w, bars):
    pt = np.array(pt, int)
    for bar in bars:
        (x, y), h = pt, w * 6
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
        if bar == 0:
            y = pt[1] + w * 2
            h = w * 2
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)

c = 200
r, sr, c2, c4 = c // 2, c // 2, c * 2, c * 4
img = np.full((c4, c4, 3), 255, np.uint8)
blue, red = ((255, 0, 0), (0, 0, 255))

cv2.ellipse(img, (c2, c2), (r, r), 0, 0, 180, blue, -1)
cv2.ellipse(img, (c2, c2), (r, r), 180, 0, 180, red, -1)
cv2.ellipse(img, (c2, c2), (r, r), 180, 0, 180, blue, -1)
cv2.ellipse(img, (c2, c2), (r, r), 0, 0, 180, red, -1)

left = (c2 - c * (18 + 8) / 24, c2 - sr)
right = (c2 + c * (18 + 0) / 24, c2 - sr)

draw_bar(img, left, c // 12, (1, 1, 1))
draw_bar(img, right, c // 12, (0, 0, 0))

# 회전 각도 계산
angle = cv2.fastAtan2(2, 3)

# 이미지 회전
img = cv2.warpAffine(img, cv2.getRotationMatrix2D((c2, c2), -angle * 2, 1), (c4, c4))

# 결과 이미지에서 태극기 부분을 잘라냄
result_img = img[c4 - c * 2:c4 + c * 2, c4 - c * 3:c4 + c * 3]

# 결과 이미지 출력
cv2.imshow('img', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
