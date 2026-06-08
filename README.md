# Serveillance_Cam_and_Alarm
A python project using Numpy, OpenCV, Threading, and a Caffe module for human detection. Created to set up cameras on a large property surrounding a home for humans and alerting the owner if they are approaching the home at Non-Visiting hours. It does this by placing each of the cameras on a unique Thread, allowing them to process simultaneously. To reduce CPU usage, the camera will look for movement before doing any detection. The detection is isolated to each individual image, creating a bounding box around the detection of a person, as long as the confidence level is above a certain threshold. If a person is detected, the system plays an alarm, alerting the user, as well as capturing an image of the person found at given intervals.

The images are then merged into a single image and shown on screen to the user.

# Modules Used:
- OpenCV
- Threading
- Numpy

# Cameras and Hardware Used:
- Rapsberry Pi 4
- Arducam 1080P Day/Night Vision USB Camera (B0506)
- Amazon Basics Stereo

# Updates:
V 0.3.1:
  - Additions:
    - Image Capture when a Person is detected
    - When a person is not detected for 5 seconds, returns to no detection state
    - Alarm sounds when a person is initially detected
  - Minor Tweaks:
    - Confidence level increased 
