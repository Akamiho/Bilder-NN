import pickle 
import pandas as pd

with open("Daten/meta","rb") as pi:
    dict = pickle.load(pi, encoding='latin1')["fine_label_names"]

def train():
    #jede der Dateien wird geladen und zu zwei Listen zusammengefasst
    with open("Daten/train","rb") as pi:
        data = pickle.load(pi, encoding='latin1') 

    return anpassen(data["data"],data["fine_labels"])

def test():
    
    #Test-Daten werden geladen
    with open("Daten/test","rb") as pi:
        data = pickle.load(pi, encoding='latin1') 
    
    #Test-Daten werden angepasst
    return anpassen(data["data"],data["fine_labels"])


#Funktion normalisiert Input, bringt in ins richige Format und macht den Output zu Kategorischen Daten
def anpassen(I,O):
    X = I / 255.0
    X = X.reshape((len(X), 3, 32, 32))
    X = X.transpose(0, 2, 3, 1)
    y = pd.get_dummies(O,columns=dict,dtype=int).to_numpy()
    
    return [X,y]