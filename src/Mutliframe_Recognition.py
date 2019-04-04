#!/usr/bin/env python
# coding: utf-8

# In[56]:


import os
import cv2
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy import misc
from Detection import Detection
from Recognition import Recognition
import preprocess
import statistics
from collections import Counter
import imutils


# In[75]:


#Path for reading the video
Input_video_path = "../../EximiusAI_Dataset/Original/Videos/20190312_153102.mp4"
Output_video_path = "../Outputs/Detection_Recognition_outputs/"
Output_video_name = "EximiusAI_10.avi"
classifier_pickle_file = "../util/EximiusAI_A.pkl"
O_path = Output_video_path + Output_video_name


# In[59]:


li = Counter(['santosh', 'santosh','Ravi','Ravi','sudheep'])
da = li.most_common(1)
print(da[0][0])


# In[60]:


#Path for the detection and recognition pb files
Detection_model_path = "../Models/Detection_mtcnn.pb"
Recognition_model_path = "../Models/Recognition_facenet.pb"


# In[61]:


#Instances of detection and recognition are being created.
#Instances are created to avoid loading the graphs and sessions again and again for every frame.
detection = Detection(Detection_model_path)
recognition = Recognition(Recognition_model_path)


# In[62]:


with open(classifier_pickle_file, 'rb') as infile:
    model,class_names = pickle.load(infile)


# In[76]:


#Initializing video capture from the Input_video_path.
cap = cv2.VideoCapture(Input_video_path)
#Variable to count frames.
frame_count = 0
#starting the processing time to calculate fps.
start = time.time()
#Ensuring the input_video_path opens without errors.
if (cap.isOpened()== False):
    print("Error opening video stream or file")

#getting the frame_width , frame_height from the given input video.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_area = frame_width * frame_height

#Change Orientation
temp = frame_height
frame_height = frame_width
frame_width = temp

#creating a video file to write the output frames at output_video_path(O_path).

out = cv2.VideoWriter(O_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30 , (frame_width,frame_height))

#Reading each and every frame in a while loop and processing/inferencing them through two models.
grab = 0
buff_count = 0
class_name = " "
frame_names = []
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = imutils.rotate_bound(frame,90)
    frame_start_time = time.time()
    frame_count = frame_count+1
    if ret != True or frame_count>10000:
        break
    if ret == True:
        mode = ""
        frame_recognition_rate = 6
        trigger = 0
        if len(frame_names) == frame_recognition_rate:
            trigger = 1
            mode = (Counter(frame_names).most_common(1))[0][0]
            frame_names = []
        
        #Inferencing the Detection model
        bbox, scores, landmarks = detection.detect(frame)
        if len(bbox) == 0:
            buff_count = buff_count + 1
        if buff_count == 3:
            frame_names = []
            mode = ""
            class_name = ""
            trigger = 1
            
        if len(bbox) == 1:
            buff_count = 0
            for box, pts in zip(bbox, landmarks):
                box = box.astype('int32')
                box_w = box[3] - box[1]
                box_h = box[2] - box[0]
                box_a = box_w*box_h
                percent = box_a*100/frame_area

                # CROPPING THE FACES OUT OF THE IMAGE AND APPENDING THEM TO THE LIST
                print('[INFO] percentage of bounding box in total image : {:.2f}'.format(percent))
                face = np.copy(frame[box[0]:box[2] , box[1]:box[3]])
                if percent >1.0 and face.shape[0] != 0 and face.shape[1]!= 0 and face.shape[2] !=0:
                    if grab == 0:
                        img = face
                    grab = grab+1
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = preprocess.prewhiten(face)
                    embedding = recognition.recognize(face = face)
                    predictions = model.predict_proba(embedding)
                    prediction_id = model.predict(embedding)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    print(best_class_probabilities)
                    prob_color_g = int(best_class_probabilities*255)
                    prob_color_r = 255 - prob_color_g
                    tup = (0,prob_color_g,prob_color_r)
                    conf = ""
                    if best_class_probabilities > 0.8:
                        conf = "high"
#                         tup = (0,255,0)
                    if best_class_probabilities > 0.5 and best_class_probabilities < 0.8:
                        conf = "low"
#                         tup = (0,255,255)
                    if best_class_probabilities < 0.5:
                        prediction_id = ["Unknown"]
#                         tup = (0,0,255)
                        conf = "-"
                    print(str(prediction_id[0]))
                    frame = cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), tup, 3)
                    pts = pts.astype('int32')
                    frame_names.append(prediction_id[0])
                    if(trigger == 1):
                        class_name = mode
                        print("TRIGGERED!!!")
                    print("Present Frame:",prediction_id[0])
                    print("Bounding Text:",class_name)
                    cv2.putText(frame, class_name, (box[1], box[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, tup ,
                                thickness=2, lineType=2)

        else :
            cv2.putText(frame, "Multiple Faces", (0, 0),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) ,thickness=2, lineType=2)
            frame_names = []
        show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         plt.imshow(show)
#         plt.show()
        out.write(frame)
        
        frame_end_time = time.time()
        time_per_frame = frame_end_time - frame_start_time
        fps_frame = 1/time_per_frame
        print('[INFO] total boxes:', len(bbox))
        print('[INFO] Processing Frame:', frame_count)
        print('[INFO] Processing Speed:',fps_frame," FPS")
        print('[INFO] Time Per Frame:', time_per_frame)
        
end = time.time()
timet = end - start
fps = frame_count/timet
print("[INFO] NUMBER OF FRAMES:", frame_count)
print("[INFO] Detection took {:.5} seconds".format(end - start))
print("[INFO] Overall FPS: "+ str(fps))

# closing the writer and reader

cap.release()
out.release()


# In[31]:


out.release()


# In[ ]:




