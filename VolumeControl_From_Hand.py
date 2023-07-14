import cv2
import mediapipe as mp
import time
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Camera Access
cap = cv2.VideoCapture(0)

# Using hand Detection Module Builtin
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # object of hand
mpDraw = mp.solutions.drawing_utils

# Camera properties
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)

# Volume control variables
minVol = 0  # Minimum volume (0%)
maxVol = 100  # Maximum volume (100%)
volBar = 400  # Length of the volume bar
vol = 0  # Current volume level
volPercentage = 0  # Current volume percentage

# Function to set the system volume
def set_system_volume(volume_level):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevelScalar(volume_level, None)

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

cTime = 0
pTime = 0

# Calling Camera
while True:
    success, img = cap.read()

    # Image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  # Process the frame of hands

    # Extract the multiple Hands
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            # Access each 20 points separately
            lmList = []
            for id, lm in enumerate(handlms.landmark):
                # This gives the each point(id) and there landmark(x,y,z) location
                # x,y,z is ratio so we convert into the pixel by multiplying it with width and height
                h, w, c = img.shape  # C for column
                cx, cy = int(lm.x * w), int(lm.y * h)  # Centre point
                lmList.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            # Drawing the 21 points with the mediaPipe
            # handlms is hand landmarks for each hand
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

            if len(lmList) != 0:
                # Get coordinates of thumb and index finger tips
                thumb_tip_x, thumb_tip_y = lmList[4][1], lmList[4][2]
                index_tip_x, index_tip_y = lmList[8][1], lmList[8][2]

                # Calculate distance between thumb and index finger tips
                distance = calculate_distance(thumb_tip_x, thumb_tip_y, index_tip_x, index_tip_y)

                # Map the distance to volume range
                vol = np.interp(distance, [20, 250], [minVol, maxVol])
                volBar = np.interp(distance, [20, 250], [400, 150])
                volPercentage = np.interp(distance, [20, 250], [0, 100])

                # Set the system volume
                set_system_volume(vol/100.0)

                # Draw volume bar and percentage
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPercentage)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
