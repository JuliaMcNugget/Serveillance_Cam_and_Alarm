import cv2 as cv
import threading

class camera():
    def __init__(self, src):
        # establishes the calls allowed to be made for each camera
            # Establish the camera
        self.capture = cv.VideoCapture(src, cv.CAP_DSHOW)
            # Establish the returned information as callable
        self.ret, self.frame = self.capture.read()
            # If initiated, the camera is started
        self.stopped = False
            # Initiates the individual camera thread
        self.lock = threading.Lock()

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
                self.stop()
                return
                # Using the threading
            with self.lock:
                    # assign the confirmation to the object
                self.ret = ret
                    # Do the same with the image
                self.frame = frame

    def read(self):
            #Return the stored image
        with self.lock:
            
            if self.frame is None:
                return None
                
            frame = self.frame.copy()

        return frame

    def stop(self):
            # End the camera function
        self.stopped = True
            # Release the camera
        self.capture.release()



