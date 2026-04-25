import cv2 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque

HAND_LANDMARKER_PATH = 'hand_landmarker.task'
BUFFER_LENGTH = 30

class SequenceBuffer:
    def __init__(self, max_length):
        self.buffer = deque(maxlen= max_length)

    def add(self, feature_vector):
        self.buffer.append(feature_vector)

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen

    def get_sequence(self):
        return np.array(self.buffer,dtype=np.float32)

    def length(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

class HandTracker:
    def __init__(self,model_path):
        self.model_path = model_path

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=1,min_hand_detection_confidence=0.5,min_hand_presence_confidence=0.5,min_tracking_confidence=0.5)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self,frame):
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=frame)
        return self.landmarker.detect(mp_image)

    def draw_landmarks(self,frame,landmarks):
        h,w,_ = frame.shape

        for landmark in landmarks:
            x = int(landmark.x*w)
            y = int(landmark.y*h)
            cv2.circle(frame,(x,y),5,(255,0,0),-1)

def landmarks_to_array(hand_landmarks):
    coords = []
    for landmark in hand_landmarks:
        coords.append([landmark.x,landmark.y,landmark.z])

    return np.array(coords,dtype=np.float32)

def normalize_landmarks(landmarks):
    landmarks = landmarks.copy()

    wrist = landmarks[0]
    landmarks = landmarks - wrist

    scale = np.linalg.norm(landmarks[9])
    if scale <= 0.000001:
        scale = 0.000001

    return (landmarks / scale)

def make_feature_vector(curr_landmarks,prev_landmarks):
    curr_norm = normalize_landmarks(curr_landmarks)
    
    if prev_landmarks is None:
        prev_norm = curr_norm
    else:
        prev_norm = normalize_landmarks(prev_landmarks)
    
    velocities = curr_norm - prev_norm

    feature_vector = np.concatenate([curr_norm.flatten(),velocities.flatten()])

    return feature_vector

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERR: webcam err")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    tracker = HandTracker(HAND_LANDMARKER_PATH)
    buffer = SequenceBuffer(BUFFER_LENGTH)
    prev_landmarks = None

    while True:
        ret,frame = cap.read()
        if not ret:
            print('ERR: failed to read video capture')
            break
        
        frame = cv2.flip(frame,1)

        result = tracker.detect(frame)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            tracker.draw_landmarks(frame,hand_landmarks)
            
            curr_landmarks = landmarks_to_array(hand_landmarks)
            feature_vector = make_feature_vector(curr_landmarks,prev_landmarks)
            buffer.add(feature_vector)
            prev_landmarks = curr_landmarks.copy()
        else:
            prev_landmarks = None


        cv2.imshow("Hand Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()