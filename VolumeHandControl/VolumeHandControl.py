import HandTrackingModule as htm
import cv2
import time
import numpy as np
import math

#https://github.com/AndreMiras/pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(0, None)
minVol=volRange[0]
maxVol=volRange[1]
volume_level=0
volume_bar=0
volume_per=0


####################################
wCam, hCam = 720, 488
####################################
cap = cv2.VideoCapture(1)
# cap.set(3,wCam)
# cap.set(4,hCam)
pTime = 0
cTime = 0
detector = htm.HandDetector(detectionConf=0.7)
while True:
    success, img = cap.read()
    hands=detector.GetHands(img, draw=True)
    finger_tip_points=detector.getPositioninPixels(img, htm.KEYPOINTS.kINDEX_FINGER_TIP,draw=True)
    thumb_tip_points=detector.getPositioninPixels(img, htm.KEYPOINTS.kTHUMB_TIP,draw=True)
    if finger_tip_points and thumb_tip_points:
        x1,y1=thumb_tip_points[0],thumb_tip_points[1]
        x2,y2=finger_tip_points[0],finger_tip_points[1]
        cx,cy=(x1+x2)//2,(y1+y2)//2
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        
        length=math.hypot(x2-x1,y2-y1)
        print(length)
        volume_level=np.interp(length,[20,120],[minVol,maxVol])
        volume_per=np.interp(volume_level,[minVol,maxVol],[0,100])
        volume_bar=np.interp(volume_level,[minVol,maxVol],[488,158])
        volume.SetMasterVolumeLevel(volume_level, None)

        if(length<50):
            cv2.circle(img,(cx,cy),10,(0, 255, 255),cv2.FILLED)
        elif(length>120):
            cv2.circle(img,(cx,cy),10,(0, 0, 255),cv2.FILLED)
            
        cv2.rectangle(img,(50,150),(85,488),(0,255,0),3)
        cv2.rectangle(img,(50,int(volume_bar)),(85,488),(0,255,0),cv2.FILLED)
        cv2.putText(img,f"{int(volume_per)} %",(48,58),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f"{str(int(fps))} fps", (0, img.shape[0]-30),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
