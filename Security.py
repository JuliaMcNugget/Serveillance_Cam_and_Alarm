import cv2 as cv
import numpy as np
from camera import camera
from datetime import datetime
import os, sys, time, pygame
#---------------------- Audio -------------------------------- 
pygame.mixer.init()
pygame.mixer.music.load("hey.mp3")
#---------------------- Audio -------------------------------- 


# --------------------- File Path Helper ---------------------

def resource_path(relative_path):
    # get the absolute path
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
        # Initiate cameras
        
#---------------------- Camera Set up ----------------------------
cams = [camera(0).start(), camera(1).start()]#, camera(10).start(), camera(13).start()]
#------------------- Camera Set Up END ---------------------------

#---------------------- Image Merging ----------------------------
#-------------------- Image Merging END --------------------------
def merge(frames):
    #if len (frames) % 2:
    combined = np.hstack((frames[0], frames[1]))
    """
    if len(frames) % 4:
        bottom = np.hstack((frames[2], frames[3]))
        combined = np.vstack((combined, bottom))
    """
    return combined
    

    # --------------------- Load Model ---------------------------

proto = resource_path("MobileNetSSD_deploy.prototxt")
model = resource_path("MobileNetSSD_deploy.caffemodel")
box = []
net = cv.dnn.readNetFromCaffe(proto, model)
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
    seen = [0] *len(cams)
    person = [False] *len(cams)
    err_message = ""
    prev_err_message = ""
    combined = None
    frame_count = 0
    img_num = [0] * len(cams)
    while True:
            # Collect frames into a single array
        
        frames = []
            # for every camera, read the information
        
        for i, cam in enumerate(cams):
            if time.time() - seen[i] > 5:
                person[i] = False
            ret, img = cams[i].capture.read()
            
            if not ret or img is None:
                err_message = f"Camera {i+1} read failed"
                continue
            # Pull the dimmensions of the image
            frame = cv.resize(img, (351, 286))
            
            h, w = frame.shape[:2]
            
            if cam.movement:
                

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
                                # create the message
                                
                                seen[i] = time.time()

                                if not person[i]:
                                    person[i] = True
                                    pygame.mixer.music.play()
                                    cv.imwrite(f"person_cam{i}_{img_num[i]}.jpg", img)
                                    img_num[i] +=1
                                    print(f"Person at camera {i+1}")
                                # Check if already existing
                                
                                    # Establish the bounds of the person detected
                                box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
        
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
        combined = merge(frames)
        
        if err_message != prev_err_message:
            prev_err_message = err_message
            if err_message != "":
                print(err_message)
            # Show on screen
        cv.imshow("Security Camera", combined)
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
