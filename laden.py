import tensorflow as tf
import pickle 
import numpy as np

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
    X = np.array(X)
    X = X / 255.0
    X = X.reshape((50000, 3, 32, 32))
    # Output-Daten werden zu Array umgewandelt, dass Warscheinlichkeiten zeigt
    y = tf.keras.utils.to_categorical(y, num_classes=10)
    
    return [X,y]

def test():
    #test-Daten werden geladen
    with open("Daten/test_batch","rb") as pi:
        data = pickle.load(pi, encoding='latin1') 
    
    #Test-Daten werden angepasst
    X = data["data"] / 255
    X = X.reshape((10000, 3, 32, 32))
    y = tf.keras.utils.to_categorical(data["labels"], num_classes=10)
    
    return [X,y]

