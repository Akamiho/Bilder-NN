import pickle
import pandas as pd

with open("Daten/batches.meta","rb") as pi:
    dict = pickle.load(pi, encoding='latin1')["label_names"]

with open("Daten/test_batch","rb") as pi:
    data = pickle.load(pi, encoding='latin1') 

y = pd.get_dummies(data=data["labels"],columns=dict,dtype=int).to_numpy()

print(y[0:10])

print(data["labels"][:10])