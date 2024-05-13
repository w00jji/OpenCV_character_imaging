import cv2
import mediapipe as mp
import numpy as np

def overlay(image,x,y,w,h,overlay_image): #대상 이미지 , x ,y 좌표 ,width , height , 덮어씌울 이미지(4채널)
    alpha = overlay_image[:,:,3] # BGRA  , 4채널의 이미지 값에서 알파 값만 때온다. 투명영역만 가져온다
    mask_image = alpha.astype(np.float32) / 255 # 0~255 ->로 나누면 0~1 사이의 값 (1: 불투명 , 0: 완전)
    #(255,255) ->(1,1)
    #(255,0) -> (1,0)

    #1 - mask_image ?
    #(0,0)
    #(0,1)
    for c in range(0,3): #chnnel BGR 을 처리
        image[y-h:y+h, x-w:x+w, c] = (overlay_image[:,:,c]*mask_image) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))


# 얼굴을 찾고, 찾은 얼굴에 표시를 해주기 위한 변수 정의
mp_face_detection = mp.solutions.face_detection  # 얼굴 검출을 위한 face detection 모듈
mp_drawing = mp.solutions.drawing_utils #얼굴의 특징을 그리기 위한 drawing_utils

# 동영상 파일 열기
cap = cv2.VideoCapture('face_video.mp4')

# 이미지 불러오기
image_right_eye = cv2.imread('right_eye.png',cv2.IMREAD_UNCHANGED) #100x100
image_left_eye = cv2.imread('left_eye.png',cv2.IMREAD_UNCHANGED) # 100x100
image_nose = cv2.imread('nose.png',cv2.IMREAD_UNCHANGED) # 300x100

# Get the original width and height of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



#model_selection = 0 근거리 1은 원거리
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        try:
            success, image = cap.read()
            if not success:
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #RGB로 변환
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.detections:
                # 6개 특징 : 오른쪽 눈 , 왼쪽 눈 , 오른쪽 귀 , 왼쪽 귀 , 코 끝 부분 , 입 중심부
                for detection in results.detections:
                    #mp_drawing.draw_detection(image, detection)
                    #print(dection)
                    #특정 위치 가져오기
                    keypoints = detection.location_data.relative_keypoints
                    right_eye = keypoints[0] #오른쪽 눈
                    left_eye = keypoints[1] #왼쪽 눈
                    nose = keypoints[2] # 코 끝부분

                    h,w, _ = image.shape # height, width, channel 이미지로부터 세로가로 크기를 가져옴

                    right_eye = (int(right_eye.x*w)-20,int(right_eye.y*h)-100) # 이미지내에서 실제 좌표 (x,y)
                    left_eye = (int(left_eye.x * w)+20, int(left_eye.y * h)-100)
                    nose_tip = (int(nose.x*w), int(nose.y*h))

                    # image,x,y,w,h,overlay_image
                    overlay(image, *right_eye, 50, 50, image_right_eye)
                    overlay(image, *left_eye, 50, 50, image_left_eye)
                    overlay(image, *nose_tip, 150, 50, image_nose)


                    #양 눈에 동그라미 그리기
                   # cv2.circle(image,right_eye,50,(255,0,0),10,cv2.LINE_AA)#파란색
                   # cv2.circle(image, left_eye, 50, (0, 255, 0), 10, cv2.LINE_AA)  # 초록색
                    # 코에 동그라미 그리기
                   # cv2.circle(image, nose_tip, 75, (0, 255, 255), 10, cv2.LINE_AA) # 노란색
                   # 각 특징에 이미지 그리기
                  #image[right_eye[1]-50 : right_eye[1]+50, right_eye[0]-50 : right_eye[0]+50 ] = image_right_eye
                  #image[left_eye[1] - 50: left_eye[1] + 50, left_eye[0] - 50: left_eye[0] + 50 ] = image_left_eye
                  #image[nose_tip[1] - 50: nose_tip[1] + 50, nose_tip[0] - 150: nose_tip[0] + 150 ] = image_nose


            # Define the scale factor for resizing
            scale_factor = 0.2
            # Resize the image with the defined scale factor
            resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

            # Show the resized image
            cv2.imshow('MediaPipe Face Detection', resized_image)
            if cv2.waitKey(1) == ord('q'):
                break
        except KeyboardInterrupt:
            break

cap.release()
cv2.destroyAllWindows()

