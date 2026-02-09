# hand_wave_detection
python script for waving hand detection, need to download opencv and mediapipe, python version of 3.11.5 used 
this works by using mediapipe to mark out the different landmarks on palm for tracking such that when the landmarks pass through the lines outlined by the right or left boxes for a certain number of times within a certain duration set, it would be registered as waving
since there are different kinds of waving, when the palm is stationary within the box, the landmark is tracked to be stationary for certain amount of time and also registered as waving 
