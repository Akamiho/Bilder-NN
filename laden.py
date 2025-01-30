import tensorflow as tf
import pickle 
import numpy as np
import pandas as pd

with open("Daten/batches.meta","rb") as pi:
    dict = pickle.load(pi, encoding='latin1')["label_names"]

def train():
    y = []
    X = []
    #jede der Dateien wird geladen und zu zwei Listen zusammengefasst
    for i in range(5):
        with open("Daten/data_batch_"+str(i+1),"rb") as pi:
            data = pickle.load(pi, encoding='latin1') 
        X += data["data"].tolist()
        y += data["labels"]  
    
    #Daten werden für das Modell angepasst(zu Numpy Array mit richtigen Dimensionen und normalisiert)
    X = np.array(X, dtype=np.float32)
    X = X / 255.0
    X = X.reshape((50000, 3, 32, 32))
    X = X.transpose(0, 2, 3, 1)
    
    # Output-Daten werden zu Array umgewandelt, dass Warscheinlichkeiten zeigt
    y = pd.get_dummies(data=y,columns=dict,dtype=int).to_numpy()
    
    return [X,y]

def test():
    
    #Test-Daten werden geladen
    with open("Daten/test_batch","rb") as pi:
        data = pickle.load(pi, encoding='latin1') 
    
    #Test-Daten werden angepasst
    X = data["data"] / 255.0
    X = X.reshape((10000, 3, 32, 32))
    X = X.transpose(0, 2, 3, 1)
    y = pd.get_dummies(data=data["labels"],columns=dict,dtype=int).to_numpy()
    
    return [X,y]

