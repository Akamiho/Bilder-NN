import tensorflow as tf
from keras import layers, models



#Modell wird erstellt
def erstellen():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32,3)),
        layers.Dropout(0.2),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Dropout(0.2),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax') 
    ])


    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

#modell wird trainiert
def trainieren(modell,daten,test):
    return modell.fit(daten[0], daten[1], epochs=8,  validation_data=test )

#modell wird anhand anderer Daten getestet
def rate(modell,daten):
    test = modell.evaluate(daten[0], daten[1])
    print("Test-Loss und Test-Genauigkeit:", test)