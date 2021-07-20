import cv2
import math
import numpy as np
import mediapipe

drawingModule = mediapipe.solutions.drawing_utils
faceModule = mediapipe.solutions.face_mesh

circleDrawingSpec = drawingModule.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
lineDrawingSpec = drawingModule.DrawingSpec(thickness=1, color=(0,255,0))

with faceModule.FaceMesh(static_image_mode=True) as face:
    # Face landmarks estimation
    image = cv2.imread("./photo.jpg")
    results = face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    points=[]
    if results.multi_face_landmarks != None:
        for faceLandmarks in results.multi_face_landmarks:
            drawingModule.draw_landmarks(image, faceLandmarks, faceModule.FACE_CONNECTIONS, circleDrawingSpec,
                                         lineDrawingSpec)
            point = []
            for id, lm in enumerate(faceLandmarks.landmark):
                print(lm)
                ih,iw,ic = image.shape
                x,y = int(lm.x * iw), int(lm.y * ih)
                cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)
                point.append([x,y])

                print(id, x, y)
            points.append(point)

    cv2.imshow('Test image', image)

    #[(283번 x좌표 - 295번 x좌표) + (385번 x좌표 - 387번 x좌표) * (295번 y좌표 - 387번 y좌표)] / 2

    print((abs(point[283][0] - point[295][0])+abs(point[387][0]-point[385][0]) * abs(point[295][1]-point[387][1]))/2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
