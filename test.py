import pickle

with open("Daten/meta","rb") as pi:
    dict = pickle.load(pi, encoding='latin1')["coarse_label_names"]
    
print(dict)