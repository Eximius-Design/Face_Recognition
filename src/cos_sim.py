import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class cos_sim:
    
    def __init__(self,path1, path2):
        embs = np.load(path1)
        embs = embs.reshape(embs.shape[0],embs.shape[2])
        labels = np.load(path2)
        print("Cosine similarity Initialized")

       
    def sim(self, embd):
        li = []
        for (i,emb) in enumerate(embs):
            li.append(cosine_similarity(emb,embd))
        name = labels[np.argmax(li)]
        print(name)
        return name,max(li)
