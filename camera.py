import cv2 as cv
import threading
import numpy as np

class camera():
    def __init__(self, src):
        # establishes the calls allowed to be made for each camera
            # Establish the camera
        self.capture = cv.VideoCapture(src)
        if not self.capture.isOpened():
                print(f"Failed to open camera {src}")
                
            # Establish the returned information as callable
        self.ret, self.frame = self.capture.read()
            # If initiated, the camera is started
        self.stopped = False
            # Initiates the individual camera thread
        self.lock = threading.Lock()
        self.previous_gray = None
        self.movement = False

    def start(self):
            # Starts the individual camera thread
        thread = threading.Thread(target = self.update, daemon=True)
        thread.start()

        return self

    def update(self):
        
            # Updates the camera feed
        while not self.stopped:

                # Captures an image and confirmation
            ret, frame = self.capture.read()
                # if nothing is received
            if not ret:
                    # End the loop
                
                frame = np.zeros((351,286,3), dtype=np.uint8)
                return
                # Using the threading
            with self.lock:
                    # assign the confirmation to the object
                self.ret = ret
                    # Do the same with the image
                self.frame = frame
                    # Check for movement
                self.movement = self.check_movement(frame)

    def read(self):
            #Return the stored image
        with self.lock:
            
            if self.frame is None:
                frame = np.zeros((351,286,3), dtype=np.uint8)
                
            else:
                frame = self.frame.copy()

        return frame

    def stop(self):
            # End the camera function
        self.stopped = True
            # Release the camera
        self.capture.release()
        
    def check_movement(self, frame):
            
            # Create base image with frame in black and white
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Blur the image at a radius off 21 pixels
        gray = cv.GaussianBlur(gray, (21, 21), 0)
            # Base variable to return
        
        if self.previous_gray is not None:
            delta = cv.absdiff(self.previous_gray, gray)
            
            thresh = cv.threshold(delta, 25, 255, cv.THRESH_BINARY)[1]
            thresh = cv.dilate(thresh, None, iterations=2)
            motion_pixels = cv.countNonZero(thresh)
            
        else:
            self.previous_gray = gray
            return False
            
        if motion_pixels > 1500:
            self.previous_gray = gray
            
        return motion_pixels > 1500
        



