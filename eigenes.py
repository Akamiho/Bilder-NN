import numpy as np
from PIL import Image
import keras
import pickle

#öffnet die Datei mit den Namen der Kategorien
with open("Daten/batches.meta","rb") as pi:
    dict = pickle.load(pi, encoding='latin1')["label_names"]
    
#öffnet das Bild und passt Größe an
image = Image.open('Daten/reh.webp')
image = image.resize((32,32))

#wandelt das Bild in ein Array um und passt dieses an
array = np.asarray([image])
array = array /255

#Modell wird geladen und auf das Bild angewendet
modell = keras.models.load_model("Daten/modell.keras")
ergebnis = modell.predict(array)

#Warscheinlichkeit und Ergebnis werden gespeichert
prob = round(100 * ergebnis.max(),2)
ergebnis = dict[ergebnis.argmax()]

print("Das Bild ist mit einer Warscheinlichkeit von", prob,"% :",ergebnis,"!")
