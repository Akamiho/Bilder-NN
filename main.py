import model
import laden
import plotter

modell = model.erstellen()

traindaten = laden.train()
testdaten = laden.test()

training = model.trainieren(modell,traindaten,testdaten)
modell.save("Daten/modell.keras")

plotter.plotAccuracy(training)
plotter.plotLoss(training)


model.rate(modell,testdaten)