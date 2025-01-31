import model
import laden
import plotter

modell = model.erstellen()

traindaten = laden.train()
testdaten = laden.test()

history = model.trainieren(modell,traindaten[0],traindaten[1],testdaten)

modell.save("Daten/modell.keras")
plotter.plotLoss(history)

plotter.plotAccuracy(history)

model.rate(modell,testdaten[0],testdaten[1])