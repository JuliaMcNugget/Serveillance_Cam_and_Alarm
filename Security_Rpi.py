import cv2 as cv
import numpy as np
from camera import camera
import os
import sys

# --------------------- File Path Helper ---------------------

def resource_path(relative_path):
    # get the absolute path
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --------------------- Load Model ---------------------------

proto = resource_path("MobileNetSSD_deploy.prototxt")
model = resource_path("MobileNetSSD_deploy.caffemodel")

net = cv.dnn.readNetFromCaffe(proto, model)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# ------------------------------------------------------------

def main():
        # establish the cameras
    cams = [camera(0).start()]
    #, camera(1).start(), camera(2).start(), camera(3).start()]

    frame_count = 0

    while True:
            # Collect frames into a single array
        frames = []
            # for every camera, read the information
        for cam in cams:
            ret, frame = cam.read()

            if not ret:
                print("Camera read failed")
                break

            frame = cv.resize(frame, (351, 286))
            frames.append(frame)
            # Check every 4 frames
        if len(frames) != 4:
            continue
            # Merge images from cameras
        top = np.hstack((frames[0], frames[1]))
        bottom = np.hstack((frames[2], frames[3]))
        combined = np.vstack((top, bottom))

        blob = cv.dnn.blobFromImage(combined, 0.007843, (702, 572), 127.5)
        
            # Pull the dimmensions of the image
        h, w = combined.shape[:2]
        
            # Select the adjusted image for detection
        net.setInput(blob)
        
            # Check for detection every other frame
        if frame_count % 2 == 0:
            detections = net.forward()

        frame_count += 1

        for i in range(detections.shape[2]):
                # For every detection, pull the confidence level
            confidence = detections[0, 0, i, 2]

            if confidence > 0.2:
                    # pull the labels
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                    # Check if what is detected is a person
                if label == "person":
                        # Establish the bounds of the person detected
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                    (startX, startY, endX, endY) = box.astype("int")
                        # Draw the rectangle
                    cv.rectangle(combined, (startX, startY), 
                        (endX, endY), (0, 255, 0), 2)
                        # Select the apropriate label
                    text = f"{label}: {confidence:.2f}"
                        # Add the text to the image over the box
                    cv.putText( combined, text, (startX, startY - 10), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Show on screen
        cv.imshow("Security Camera", combined)

        key = cv.waitKey(1)

        if key == 27:
            break

    # Close all cameras
    for cam in cams:
        cam.release()

    cv.destroyAllWindows()

# ------------------------------------------------------------

if __name__ == "__main__":
    main()
