# hand wave and face detection 
python script for waving hand and face detection and the landmarks used by the hand wave detection script can be accessed via the add branch 
script opens the computer's camera and detects 2 possible hand waves which are the conventional hand wave and stationary hand and prints "hand waving" 
script can detect presence of a face and prints "face detected"
i did not train a model from scratch to be able to detect faces by using Google's pretrained lightweight face detection model, Blazeface short range, which can detect faces at a short range and is optimised for real-time detection
this is also made possible using mediapipe, openCV
the script is then implemented into a ROS script for more_robot which will be further developed to allow interaction between the robot and user where if user wave their hand, the script will allow the robot to acknowledge that there is a waving hand and then it will readjust it's head to find the user's face based on the face's coordinates 
