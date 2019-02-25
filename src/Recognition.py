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
#         with graph.as_default():
#             with tf.gfile.FastGFile(model_path, 'rb') as f:
#                 graph_def = tf.GraphDef.FromString(f.read())
#                 tf.import_graph_def(graph_def, name='')
        
#         self.graph = tf.get_default_graph()
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         self.sess = tf.Session(graph=graph, config=config)
        
   
    def recognize(self, face):
#         feeds = { self.graph.get_operation_by_name('input').outputs[0]: face, 
#                  self.graph.get_operation_by_name('phase_train').outputs[0]: False}
#         fetches = [self.graph.get_operation_by_name('embeddings').outputs[0]]
#         embeddings = self.sess.run(fetches, feed_dict=feeds)
        face = cv2.resize(face,(160,160))
        face = face.reshape(1,160,160,3)
        images_placeholder = self.graph.get_tensor_by_name("input:0")
        embeddings = self.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
        feed_dict = { images_placeholder:face, phase_train_placeholder:False }
        embeddings = self.sess.run(embeddings, feed_dict=feed_dict)
        return embeddings