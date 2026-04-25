# hands-free
Allows you to perform computer inputs by performing gestures in front of a webcam.

model
|- TCNN.pt: pytorch model of the best model achieved by the trainer

hand_tracker.py: this contains the classes and functions related to camera, feature creation,
                 and hand tracking. Main contains the test script i created to test all the 
                 features in this file.

record_data.py: this contains the script to record and save samples to a data folder. samples
                are stored in the following manner:
data
|-none
  |-sample_00.npy
  |-sample_01.npy
  |-.....
|-swipe_down
|-swipe_left
|-swipe_right
|-swipe_up

train.py: trains and evaluates the model with combinations of hyperparameters
live_predictions: This is the final product, after training. It predicts hand gestures using
                  a webcam and performs windows inputs.

hand_landmarker.task: required for the hand landmarker
