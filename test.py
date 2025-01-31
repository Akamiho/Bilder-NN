import keras
import model
import laden

modell = keras.models.load_model("Daten/modell.keras")

traindaten = laden.train()
testdaten = laden.test()

model.rate(modell,traindaten[0],traindaten[1])
model.rate(modell,testdaten[0],testdaten[1])