import model
import laden

modell = model.erstellen()

traindaten = laden.train()

model.trainieren(modell,traindaten[0],traindaten[1])
modell.save("Daten/modell.keras")

testdaten = laden.test()
model.rate(modell,testdaten[0],testdaten[1])