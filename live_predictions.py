import torch
import torch.nn.functional as f
import cv2
import numpy as np
import time
import pyautogui

from train import TemporalCNN, GESTURES, MODEL_DIR
from hand_tracker import HandTracker, landmarks_to_array,make_feature_vector,SequenceBuffer, HAND_LANDMARKER_PATH, BUFFER_LENGTH

PREDICT_EVERY_N_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.7
VOLUME_STEP = 0.05
COOLDOWN = 0.5

class Inference:
    def __init__(self,model_path,input_dim=126,num_classes=5):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print('using device: ',self.device)

        self.classes = GESTURES
        self.model = TemporalCNN().to(self.device)
        self.model.load_state_dict(torch.load('model/TCNN.pt',map_location=self.device))
        self.model.eval()

    def predict(self,sequence):
        x = torch.tensor(sequence,dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = f.softmax(logits,dim=1)
            confidence, pred_idx = torch.max(probs,dim=1)

        pred_idx = pred_idx.item()
        confidence = confidence.item()
        pred_label = self.classes[pred_idx]

        return pred_label, confidence, probs.squeeze(0).detach().cpu().numpy()

class WindowsController:
    def __init__(self):
        self.last_trigger_time = 0.0

    def can_trigger(self):
        now = time.time()
        if now > self.last_trigger_time + COOLDOWN:
            return True
        else:
            return False
        
    def execute(self, gesture):
        if self.can_trigger():
            if gesture == 'swipe_left':
                pyautogui.press('nexttrack')
            elif gesture == 'swipe_right':
                pyautogui.press('prevtrack')
            elif gesture == 'swipe_up':
                pyautogui.press('volumeup')
            elif gesture == 'swipe_down':
                pyautogui.press('volumedown')
            self.last_trigger_time = time.time()

def display_ui(frame,pred,confidence,probs,buffer_len):
    y=30
    cv2.putText(
        frame,
        f"Buffer: {buffer_len}/{BUFFER_LENGTH}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    y += 35

    cv2.putText(
        frame,
        f"Prediction: {pred}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    y += 35

    cv2.putText(
        frame,
        f"Confidence: {confidence}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    y += 55

    cv2.putText(
        frame,
        f"Probs:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    y += 25
    for i, (gesture,prob) in enumerate(zip(GESTURES,probs)):
        cv2.putText(
            frame,
            f"{gesture}: {prob}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )
        y += 25

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERR: webcam err")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    tracker = HandTracker(HAND_LANDMARKER_PATH)
    buffer = SequenceBuffer(BUFFER_LENGTH)
    inference_model = Inference(MODEL_DIR)
    controller = WindowsController()

    prev_landmarks = None
    frame_count = 0
    buffer_timer = 0.0
    predict_timer = 0.0

    current_pred = 'None'
    current_confidence = 0.0
    current_probs = np.zeros(len(GESTURES),dtype=np.float32)

    while True:
        ret,frame = cap.read()
        if not ret:
            print('ERR: failed to read video capture')
            break
        
        frame = cv2.flip(frame,1)
        result = tracker.detect(frame)

        if result.hand_landmarks:
            buffer_timer = time.time()
            if time.time() - predict_timer > COOLDOWN:
                hand_landmarks = result.hand_landmarks[0]
                tracker.draw_landmarks(frame,hand_landmarks)
                
                curr_landmarks = landmarks_to_array(hand_landmarks)
                feature_vector = make_feature_vector(curr_landmarks,prev_landmarks)
                buffer.add(feature_vector)
                prev_landmarks = curr_landmarks.copy()

                if buffer.is_full():
                    frame_count +=1

                    if frame_count % PREDICT_EVERY_N_FRAMES == 0:
                        sequence = buffer.get_sequence()
                        pred_label, confidence, probs = inference_model.predict(sequence)
                        if confidence >= CONFIDENCE_THRESHOLD:
                            current_pred = pred_label
                            if pred_label != 'none':
                                controller.execute(pred_label)
                                buffer.clear()
                                predict_timer = time.time()
                        else:
                            current_pred = 'try again'
                        current_probs = probs
                        current_confidence = confidence
        else:
            if time.time() - buffer_timer >= 2.5 and buffer.length() != 0:
                buffer.clear()
                print('buffer cleared')

            prev_landmarks = None
            current_pred = 'None'
            current_confidence = 0.0
            current_probs = np.zeros(len(GESTURES),dtype=np.float32)

        display_ui(frame,current_pred,current_confidence,current_probs,buffer.length())

        cv2.imshow("test",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()