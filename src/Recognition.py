import tensorflow as tf
import numpy as np
import cv2


class Recognition:
    
    def __init__(self, model_path):
        
        graph = tf.Graph()
        with graph.as_default():
            with open(model_path, 'rb') as f:
                graph_def = tf.GraphDef.FromString(f.read())
                tf.import_graph_def(graph_def, name='')
        self.graph = graph
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)
        print("Recognition Model Graph Initialized")

    def recognize(self, face):
# FACENET
        face = cv2.resize(face,(160,160))
        face = face.reshape(1,160,160,3)
        images_placeholder = self.graph.get_tensor_by_name("input:0")
        embeddings = self.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
        feed_dict = { images_placeholder:face, phase_train_placeholder:False }

#INSIGHT FACE
#         face = cv2.resize(face,(112,112))
#         face = face.reshape(1,112,112,3)
#         images_placeholder = self.graph.get_tensor_by_name("img_inputs:0")
#         embeddings = self.graph.get_tensor_by_name("resnet_v1_50/E_BN2/Identity:0")
#         phase_train_placeholder = self.graph.get_tensor_by_name("dropout_rate:0")
#         feed_dict = { images_placeholder:face, phase_train_placeholder:0.5 }


#Common
        
        embeddings = self.sess.run(embeddings, feed_dict=feed_dict)
        return embeddings
