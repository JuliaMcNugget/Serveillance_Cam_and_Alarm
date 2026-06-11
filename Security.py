import cv2 as cv
import numpy as np
from camera import camera
from datetime import datetime
import os, sys, time, pygame
#---------------------- Audio -------------------------------- 
pygame.mixer.init()
pygame.mixer.music.load("hey.mp3")
#---------------------- Audio END ----------------------------


# --------------------- File Path Helper ---------------------

def resource_path(relative_path):
    # get the absolute path
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
# --------------------- File Path Helper ---------------------
        # Initiate cameras
        
#---------------------- Camera Set up ----------------------------
#cams = [camera(0).start(), camera(1).start(), camera(2).start(), camera(13).start()]
cams = []

for i in range(4):
    cam = camera(i).start()
    ret, img = cam.capture.read()

    if ret and (img is not None):
        cams.append(cam)
    else:
        cam.stop()

#------------------- Camera Set Up END ---------------------------

#---------------------- Image Merging ----------------------------

def merge(frames):
    height, width, _ = frames[0].shape
        
    if len(frames) % 4 == 0:
        top = np.hstack((frames[0], frames[1]))
        bottom = np.hstack((frames[2], frames[3]))
        combined = np.vstack((top, bottom))
        position = ((width*2 - 210), (height*2 - 10))

    elif len(frames) % 3 == 0:
        top = np.hstack((frames[0], frames[1]))
        bottom = np.hstack((frames[2], np.zeros((height, width, 3), dtype = np.uint8)))
        combined = np.vstack((top, bottom))
        position = ((width*2 - 210), (height*2 - 10))
    
    elif len(frames) % 2 == 0:
        combined = np.hstack((frames[0], frames[1]))
        position = ((width*2 - 210), height - 10)
    
    
    return combined, position
    
#-------------------- Image Merging END --------------------------
    # --------------------- Load Model --------------------

proto = resource_path("MobileNetSSD_deploy.prototxt")
model = resource_path("MobileNetSSD_deploy.caffemodel")
box = []
net = cv.dnn.readNetFromCaffe(proto, model)
win_name = "Camaras de Seguridad"
cv.namedWindow(win_name, cv.WINDOW_NORMAL)
cv.setWindowProperty(win_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
print("Initiating Program")
time.sleep(2)
# ------------------------------------------------------------
def main():
        # establish the cameras
    frame_none = 0
    seen = [0] *len(cams)
    cap_time = [0] *len(cams)
    person = [False] *len(cams)
    err_message = ""
    prev_err_message = ""
    combined = None
    frame_count = 0
    img_num = [0] * len(cams)

    path = resource_path("images")

    if not os.path.exists(path):
        os.makedirs(path)

    print(f"files being saved to: {path}")
    while True:
            # Collect frames into a single array
        date = datetime.now()
        date_str = date.strftime("%Y-%m-%d")
        date_and_time = date.strftime("%Y-%m-%d_%H-%M-%S")
        frames = []

            # for every camera, read the information
        
        for i, cam in enumerate(cams):
            err_message = ""
            if time.time() - seen[i] > 5:
                person[i] = False

            ret, img = cams[i].capture.read()

                # if nothing is capterd by the ccamera, print an error message, determine that no person is identified and skip the rest of the loop
            if not ret or img is None:
                err_message = f"Camera {i+1} read failed"
                person[i] = False
                continue

                # Resize the incoming image
            frame = cv.resize(img, (351, 286))
                # Pull the dimmensions of the image
            h, w = frame.shape[:2]
            
            if cam.movement:
                
                # Create a blob from the image to be used for detection
                blob = cv.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
                
                    # Select the adjusted image for detection
                net.setInput(blob)
                
                detections = net.forward()
                
                if detections is not None:
                    
                    for j in range(detections.shape[2]):
                            # For every detection, pull the confidence level
                        confidence = detections[0, 0, j, 2]
                        
                    
                        if confidence > 0.3:
                                # pull the labels
                            idx = int(detections[0, 0, j, 1])
                            label = CLASSES[idx]

                                # Check if what is detected is a person
                            
                            if label == "person":
                                
                                    # Update the time a person was last seen
                                seen[i] = time.time()

                                    # Check if a person was detected previously
                                if not person[i]:
                                        # set detection to True, play the alarm and save an image
                                    person[i] = True
                                        # Play the alarm
                                    pygame.mixer.music.play()
                                        #Determine the date and time of the image capture
                                    date = datetime.now()
                                    date_str = date.strftime("%Y-%m-%d") 
                                    cap_time[i] = time.time()
                                    filename = os.path.join(path, f"{date_and_time}_person_cam{i}_{img_num[i]}.jpg")
                                    cv.imwrite(filename, img)
                                        # Increase the image number for the camera number
                                    img_num[i] +=1
                                        # Print that someoen was found on the camera
                                    print(f"Person at camera {i+1}")
                                    
                                    # If it's been more than a ssecond and someone is still detected, it will save another image
                                if time.time() - cap_time[i] > 2:
                                    filename = os.path.join(path, f"{date_and_time}_person_cam{i}_{img_num[i]}.jpg")
                                    cv.imwrite(filename, img)
                                    cap_time[i] = time.time()
                                    img_num[i] +=1

                                    # Establish the bounds of the person detected
                                box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                                       # Turn the info into a singular integer
                                (startX, startY, endX, endY) = box.astype("int")
                                    # Draw the rectangle
                                cv.rectangle(frame, (startX, startY), 
                                    (endX, endY), (0, 255, 0), 2)
                                    # Select the apropriate label
                                text = f"{label}: {confidence:.2f}"
                                    # Add the text to the image over the box
                                cv.putText(frame, text, (startX, startY - 10), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


            #cv.imshow("Security Camera", frame)
            frames.append(frame)
            # Check if frames exist
        if len(frames) != len(cams):
            continue
            # Merge images from cameras
        combined, position = merge(frames)
        
        if err_message != prev_err_message:
            prev_err_message = err_message
            if err_message != "":
                print(err_message)
            # Show on screen
        
        cv.putText(combined, date_and_time, position, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.imshow(win_name, combined)
        frame_count += 1
        key = cv.waitKey(1)
        #time.sleep(0.1)
        
        if key == 27:
            break

    # Close all cameras
    for cam in cams:
        cam.stop()

    cv.destroyAllWindows()

# ------------------------------------------------------------

if __name__ == "__main__":

    main()
