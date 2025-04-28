#!/usr/bin/env python3

import imutils
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import time
import numpy as np
import atexit  # Ensures cleanup on exit
from CalcAngle import CalcAngle
from Servo import *
from Motor import *

# HSV Range for Blue Lane Detection
lower_blue = np.array([81, 70, 41])
upper_blue = np.array([179, 255, 255])

# Initialize Steering and Motor
steer = Servo()
power = Motor()

# Camera Settings
camera = PiCamera()
__SCREEN_WIDTH = 320
__SCREEN_HEIGHT = 240
camera.resolution = (__SCREEN_WIDTH, __SCREEN_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(__SCREEN_WIDTH, __SCREEN_HEIGHT))

# Allow camera to warm up
time.sleep(1)

power.setMotorModel(32,32)

try:
    # Start Processing Frames
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = frame.array
        frame = imutils.rotate(frame, angle=180)
        
        # Process Image to Find Lane
        angle = CalcAngle(frame, lower_blue, upper_blue).get_angle()
        print(angle)
        steer.setServoPwm('4', angle)
        rawCapture.truncate(0)
except KeyboardInterrupt:
    steer.setServoPwm('4',90)
    camera.close()
    power.setMotorModel(0,0)
finally:
    steer.setServoPwm('4', 90)
    camera.close()
    power.setMotorModel(0.0)