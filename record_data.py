import os
import cv2 
import numpy as np
from hand_tracker import HandTracker
from hand_tracker import landmarks_to_array, make_feature_vector
from hand_tracker import SequenceBuffer


HAND_LANDMARKER_PATH = 'hand_landmarker.task'
BUFFER_LENGTH = 30

GESTURES = ['swipe_left','swipe_right','swipe_up','swipe_down','none']

def create_data_file():
    os.makedirs('data',exist_ok=True)
    for gesture in GESTURES:
        os.makedirs(os.path.join('data',gesture),exist_ok=True)

def save_datapoint(sequence,gesture):
    gesture_dir = os.path.join('data',gesture)

    existing_files = []
    for f in os.listdir(gesture_dir):
        if f.endswith('.npy'):
            existing_files.append(f)

    data_id = len(existing_files)
    file_path = os.path.join(gesture_dir,f'sample_{data_id:03d}.npy')
    np.save(file_path,sequence)
    print(f'saved: {file_path} shape = {sequence.shape}')

def display_ui(frame, current_gesture, sample_counts, buffer_len):
    y = 30

    cv2.putText(
        frame,
        f"Current label: {current_gesture}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    y += 35

    cv2.putText(
        frame,
        f"Buffer: {buffer_len}/{BUFFER_LENGTH}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2
    )
    y += 35

    cv2.putText(
        frame,
        "Keys: [1]Left [2]Right [3]Up [4]Down [5]None",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )
    y += 30

    cv2.putText(
        frame,
        "Press [s] to save current sequence | [c] clear buffer | [q] quit",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )
    y += 40

    for gesture, count in sample_counts.items():
        cv2.putText(
            frame,
            f"{gesture}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2
        )
        y += 28

def count_samples():
    counts = {}
    for gesture in GESTURES:
        gesture_dir = os.path.join('data',gesture)
        files = []
        for f in os.listdir(gesture_dir):
            if f.endswith('.npy'):
                files.append(f)
        counts[gesture]=len(files)
    return counts

def main():
    create_data_file()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERR: webcam err")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    tracker = HandTracker(HAND_LANDMARKER_PATH)
    buffer = SequenceBuffer(BUFFER_LENGTH)
    prev_landmarks = None
    current_gesture = GESTURES[0]

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

            if buffer.is_full():
                cv2.putText(
                    frame,
                    "Sequence ready to save",
                    (850, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
        else:
            prev_landmarks=None

        sample_counts = count_samples()
        display_ui(frame,current_gesture,sample_counts,buffer.length())

        cv2.imshow("Record Dataset",frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            current_gesture = "swipe_left"
        elif key == ord("2"):
            current_gesture = "swipe_right"
        elif key == ord("3"):
            current_gesture = "swipe_up"
        elif key == ord("4"):
            current_gesture = "swipe_down"
        elif key == ord("5"):
            current_gesture = "none"

        elif key == ord("c"):
            buffer.clear()
            prev_landmarks = None
            print("Buffer cleared")

        elif key == ord("s"):
            if buffer.is_full():
                sequence = buffer.get_sequence()
                save_datapoint(sequence, current_gesture)
            else:
                print("Save when buffer is full")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()