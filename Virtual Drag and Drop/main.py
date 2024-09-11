import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Set up the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height

# Initialize the HandDetector
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)
cx, cy, w, h = 100, 100, 200, 200


class DragRect():
    def __init__(self,posCenter,size=[200,200]):
        self.posCenter = posCenter
        self.size = size

    def update(self,cursor):
        cx,cy = self.posCenter
        w,h = self.size
        #if the index fingre tip is in the rectangle  region
        if cx - w // 2 < cursor_x < cx + w // 2 and cy - h // 2 < cursor_y < cy + h // 2:
            self.posCenter = cursor_x, cursor_y  # Move the rectangle with the cursor

rectList=[]
for x in range(5):
    rectList.append(DragRect([x*250+150,150]))





while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally for mirror view
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        # Get the first hand detected
        hand1 = hands[0]  # Only consider the first hand
        lmList = hand1["lmList"]  # Landmark list for the first hand

        # Get coordinates of landmarks 8 and 12
        p1 = lmList[8]  # Landmark 8 coordinates
        p2 = lmList[4]  # Landmark 12 coordinates

        # Calculate distance between these landmarks manually
        distance = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        # Draw circles for landmarks and distance
        cv2.circle(img, (int(p1[0]), int(p1[1])), 10, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (int(p2[0]), int(p2[1])), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)

        # Get the index tip (landmark 8)
        index_tip = lmList[8]
        cursor_x, cursor_y, _ = index_tip  # Use _ to ignore the z coordinate
        cv2.circle(img, (int(cursor_x), int(cursor_y)), 10, (0, 255, 0), cv2.FILLED)

        if distance < 40:
            # call the update here
            for rect in rectList:
                rect.update(cursor_x)
                rect.update(cursor_y)
    # Draw the rectangle on the image Solid
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(img,(cx - w // 2, cy - h // 2,  w ,  h) ,20,rt=0)



    # Show the processed image
    cv2.imshow('Image', img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
