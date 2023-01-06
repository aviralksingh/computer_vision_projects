import cv2
import mediapipe
import time
from enum import Enum

class KEYPOINTS(Enum):
    kWRIST=0
    kTHUMB_CMC=1
    kTHUMB_MCP=2
    kTHUMB_IP=3
    kTHUMB_TIP=4
    kINDEX_FINGER_MCP=5
    kINDEX_FINGER_PIP=6
    kINDEX_FINGER_DIP=7
    kINDEX_FINGER_TIP=8
    kMIDDLE_FINGER_MCP=9
    kMIDDLE_FINGER_PIP=10
    kMIDDLE_FINGER_DIP=11
    kMIDDLE_FINGER_TIP=12
    kRING_FINGER_MCP=13
    kRING_FINGER_PIP=14
    kRING_FINGER_DIP=15
    kRING_FINGER_TIP=16
    kPINKY_FINGER_MCP=17
    kPINKY_FINGER_PIP=18
    kPINKY_FINGER_DIP=19
    kPINKY_FINGER_TIP=20

class HandDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionConf=0.5, trackConfidence=0.5):
        self.mpHands = mediapipe.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=mode,
               max_num_hands=maxHands,
               model_complexity=model_complexity,
               min_detection_confidence=detectionConf,
               min_tracking_confidence=trackConfidence)
        self.mpDraw = mediapipe.solutions.drawing_utils


    def GetHands(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.hand_list = self.hands.process(imgRGB)

        if self.hand_list.multi_hand_landmarks:
            for handlns in self.hand_list.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handlns, self.mpHands.HAND_CONNECTIONS)
        return self.hand_list

    def getPosition(self, img, keypoint=KEYPOINTS.kWRIST,handno=0):
        ret=[]
        if self.hand_list.multi_hand_landmarks:
            myHand=self.hand_list.multi_hand_landmarks[handno]   
            ret.append(int(myHand.landmark[keypoint.value].x), int(myHand.landmark[keypoint.value].y))
        return ret

    def getPositioninPixels(self, img, keypoint=KEYPOINTS.kWRIST,handno=0,draw=True):
        ret=()
        if self.hand_list.multi_hand_landmarks:
            myHand=self.hand_list.multi_hand_landmarks[handno]           
            # print(id, ln)
            # land marks are in ratio of pixel to image dimensions. ln=px/w, py/h
            h, w, c = img.shape
            ret = (int(myHand.landmark[keypoint.value].x*w), int(myHand.landmark[keypoint.value].y*h))
            if draw:
                cv2.circle(img, (ret[0], ret[1]), 10,
                           (255, 255, 0), cv2.FILLED)
        return ret


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    pTime = 0
    cTime = 0
    detector = HandDetector()
    while True:
        success, img = cap.read()
        detector.GetHands(img,draw=True)
        detector.getPositioninPixels(img)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
