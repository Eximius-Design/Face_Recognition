import tensorflow as tf
import numpy as np
import cv2
import os
import pickle
import preprocess

def classify_image(output_classifier_path ,detection , recognition ,test_image_input_path):
    with open(output_classifier_path, 'rb') as infile:
        model ,class_names = pickle.load(infile)
    image = cv2.imread(test_image_input_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bbox, scores, landmarks = detection.detect(rgb_image)
    for box in bbox:
        box = box.astype('int32')
        face = np.copy(rgb_image[box[0]:box[2] , box[1]:box[3]])
        processed_face = preprocess.prewhiten(face)
        emb = recognition.recognize(processed_face)
        predictions_proba = model.predict_proba(emb)
        predicted_name = model.predict(emb)
        print(predicted_name, "   ", predictions_proba)