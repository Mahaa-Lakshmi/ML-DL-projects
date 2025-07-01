import cv2
import argparse
import sys

#path to prototext
path_to_ptext="ComputerVision/Models/deploy.prototxt"
path_to_model="ComputerVision/Models/res10_300x300_ssd_iter_140000.caffemodel"

#set video capture source
s=0
if len(sys.argv)>1:
    s=sys.argv[1]

#create video capture object
source=cv2.VideoCapture(s)

#create preview Window
win_name="Camera Preview"
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)

#create a new instancce of neaural network
net=cv2.dnn.readNetFromCaffe(path_to_ptext,path_to_model)

#model parameters
in_width= 300
in_height= 300
mean= [104, 177, 123]
conf_threshold=0.7

while cv2.waitKey(1)!=27: #until ESC key is pressed
    has_frame,frame=source.read()

    # check if there is no frame
    if not has_frame:
        break

    #flip the video horizontally
    frame=cv2.flip(frame,1)

    #get height and width of the frame
    frame_height,frame_width=frame.shape[0],frame.shape[1]

    #pre-process the image
    blob=cv2.dnn.blobFromImage(frame,1,(in_width,in_height),mean,swapRB=False,crop=False)

    #detections
    net.setInput(blob)
    detections=net.forward()

    #annotate the detections and loop over then
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>=conf_threshold:
            #get bounding box co-ordinates
            x_start=int(detections[0,0,i,3]*frame_width)
            y_start=int(detections[0,0,i,4]*frame_height)
            x_end=int(detections[0,0,i,5]*frame_width)
            y_end=int(detections[0,0,i,6]*frame_height)

            #object bounding box
            cv2.rectangle(frame,(x_start,y_start),(x_end,y_end),(0,255,0))

            #prepare the label
            label="Confidence: %.4f" %confidence

            #get bounding box dimensions
            label_size,baseline = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)

            #draw rectangle for the text
            #cv2.rectangle(frame,(x_start,y_start-label_size[1]),(x_end-label_size[0],y_end+baseline),(255,255,255))

            cv2.putText(frame,label,(x_start,y_start),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))

        #display inference time
        t,_=net.getPerfProfile()

        #get interal label
        label="Infernce time %.2f ms" %((t*1000.0) /cv2.getTickFrequency())

        #display label
        cv2.putText(frame,label,(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        #display the frame
        cv2.imshow(win_name,frame)

#close the video capture devices
source.release()

#destroy the window
cv2.destroyWindow(win_name)

