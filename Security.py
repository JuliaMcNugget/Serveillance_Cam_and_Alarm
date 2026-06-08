import cv2 as cv
import numpy as np
import pyautogui
import tkinter as tk
import threading

#-------------------------- Set Up Detector Info ---------------------------------------------
net = cv.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#-------------------------- Detector Set Up END ----------------------------------------------
# Screen shot info

region = (10,256,1930,1000)
screenshot = pyautogui.screenshot(region=region)
img_np = np.array(screenshot)
frame = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
width = int(frame.shape[1])
height = int(frame.shape[0])

# end screen shot info
#-------------------------- Window Set Up ----------------------------------------------------

window = tk.Tk()

#-------------------------- Window END -------------------------------------------------------

# -------------------------Text Setup --------------------------------------------------------
# Font settings
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
text_color = (255, 0, 0)  # Red
#-------------------------- Text END ---------------------------------------------------------

# Modules set up END -------------------------------------------------------------------------
kernel = (5,5)
output_filename = "Person_"


# --------------------------------------------------------------------------------------------
#-------------------------- Initializations --------------------------------------------------
frame_count = 0

#-------------------------- Main Loop --------------------------------------------------------

while True:

    #---------------------- Screen Shot Conversion -------------------------------------------
        # Receive Image    
    screenshot = pyautogui.screenshot(region=region)
        # convert to appropriate array
    img_np = np.array(screenshot)
        #convert to appropriate color space
    img = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
    #---------------------- Screen Shot Conversion END ---------------------------------------


    #---------------------- Person Detection -------------------------------------------------
    h, w = img.shape[:2]
    detect = cv.dnn.blobFromImage(cv.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(detect)
    if frame_count % 2 == 0:
        detections = net.forward()
    frame_count += 1
    for i in range(detections.shape[2]):
        confidence = detections [0, 0, i, 2]

        # Only consider detections with confidence above a certain threshold
        if confidence > 0.2:

            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            # if someone is detected
            if label == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw bounding box and label on the image
                cv.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                text = f"{label}: {confidence:.2f}"
                cv.putText(img, text, (startX, startY - 10), font, font_scale, text_color, font_thickness)
    #----------------------- Person Detection END --------------------------------------------

    #----------------------- Display Image ---------------------------------------------------
    cv.imshow("Security Camera", img)
    #----------------------- Display Image END -----------------------------------------------

    #----------------------- Exit Condition --------------------------------------------------
    k = cv.waitKey(1)  # wait a milisecond

    if k==27: # if escape it ends the program
        break
    #----------------------- Exit Condition END ----------------------------------------------
    
cv.destroyAllWindows()
#-------------------------- Main END ---------------------------------------------------------