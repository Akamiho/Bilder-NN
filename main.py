import model
import laden

modell = model.erstellen()

for i in range(5):
    data = laden.laden(i+1)
    model.trainieren(modell,data[0],data[1])
    modell.save("Daten/modell.keras")
    print("trainiert an set ",i+1)
