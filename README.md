Air-Drawing Virtual Whiteboard

Project Description
This project is a virtual whiteboard that allows users to draw on a screen using hand gestures detected through a webcam. The application was developed using Python, OpenCV, and MediaPipe.

Code Origin and Resources
The application was written by me using information learned from online documentation and tutorials.

1. LearnOpenCV Tutorial
https://docs.opencv.org/4.13.0/dd/d43/tutorial_py_video_display.html
I used this tutorial to learn the basics of working with images and OpenCV, which helped me implement the webcam input and display.

2. MediaPipe Hand Landmarker Documentation
https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
I used this documentation to learn how to detect hands and access hand landmarks using MediaPipe. This allowed me to track finger positions and develop the gesture controls used in the application.

## Features
Hand and finger tracking using MediaPipe
Drawing and erasing using pinch gestures
Gesture-controlled menu
Brush size and color selection
Undo and clear functionality

## Technologies Used
Python
OpenCV
MediaPipe
NumPy

## How to Run
1. Install Python and the required libraries.
2. Download the MediaPipe hand landmarker model if it
   is not included in the repository.
3. Make sure a webcam is connected.
4. Run the program using:
   python main.py

## Controls
- Peace sign: Open or close the radial menu
- Pinch: Draw, erase, or select a menu option
- Q: Quit the application