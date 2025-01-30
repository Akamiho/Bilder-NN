import pickle 
import pandas as pd

with open("Daten/meta","rb") as pi:
    dict = pickle.load(pi, encoding='latin1')["fine_label_names"]

def train():
    #jede der Dateien wird geladen und zu zwei Listen zusammengefasst
    with open("Daten/train","rb") as pi:
        data = pickle.load(pi, encoding='latin1') 
    
    X = data["data"]
    y = data["fine_labels"]
    #Daten werden für das Modell angepasst(zu Numpy Array mit richtigen Dimensionen und normalisiert)
    X = X / 255.0
    X = X.reshape((50000, 3, 32, 32))
    X = X.transpose(0, 2, 3, 1)
    
    # Output-Daten werden zu Array umgewandelt, dass Warscheinlichkeiten zeigt
    y = pd.get_dummies(data=y,columns=dict,dtype=int).to_numpy()
    
    return [X,y]

def test():
    
    #Test-Daten werden geladen
    with open("Daten/test","rb") as pi:
        data = pickle.load(pi, encoding='latin1') 
    
    #Test-Daten werden angepasst
    X = data["data"] / 255.0
    X = X.reshape((10000, 3, 32, 32))
    X = X.transpose(0, 2, 3, 1)
    y = pd.get_dummies(data=data["fine_labels"],columns=dict,dtype=int).to_numpy()
    
    return [X,y]

