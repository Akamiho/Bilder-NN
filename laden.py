import pickle 
import numpy as np
import pandas as pd

#Datei mit Bezeichnungen wirdgeladen
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

    return anpassen(np.array(X, dtype=np.float32),y)


def test():
    
    #Test-Daten werden geladen
    with open("Daten/test_batch","rb") as pi:
        data = pickle.load(pi, encoding='latin1') 
    
    #Test-Daten werden angepasst    
    return anpassen(data["data"],data["labels"])

#Funktion normalisiert Input, bringt in ins richige Format und macht den Output zu Kategorischen Daten
def anpassen(I,O):
    X = I / 255.0
    X = X.reshape((len(X), 3, 32, 32))
    X = X.transpose(0, 2, 3, 1)
    y = pd.get_dummies(O,columns=dict,dtype=int).to_numpy()
    
    return [X,y]