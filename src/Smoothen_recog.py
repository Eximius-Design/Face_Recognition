from collections import Counter
import numpy as np
import tensorflow as tf
from Recognition import Recognition
from sklearn.svm import SVC
import pickle

class Smoothen_recog:
    
    def __init__(self, model_path, classifier_path):
        self.recognition = Recognition(model_path)
        self.buff_count = 0
        self.class_name = ""
        self.frame_names = []
        with open(classifier_path, 'rb') as infile:
            self.model,self.class_names = pickle.load(infile)
        
    def recog(self,status,face):
        
        name = ""
        prob = 0
        print(self.frame_names)
        
        if status == 0:
            self.buff_count = self.buff_count + 1
            if self.buff_count == 4:
                frame_names = []
                self.class_name = ""
        if status == 1:
            self.buff_count = 0
            embedding = self.recognition.recognize(face = face)
            name = self.model.predict(embedding)
            prob = np.max(self.model.predict_proba(embedding))
            print("PROB:",prob)
            if (prob < 0.5):
                name[0] = "Unknown"
            self.frame_names.append(name[0])
            if len(self.frame_names) == 6:
                self.class_name = (Counter(self.frame_names).most_common(1))[0][0]
                self.frame_names = []
       
        if status >= 2:
            self.frame_names = []
        
        return [self.class_name], prob
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            
            