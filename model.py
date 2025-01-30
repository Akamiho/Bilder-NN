import tensorflow as tf
from keras import layers, models
import pickle



# Daten laden und festlegen
with open("Daten/data_batch_1","rb") as pi:
    data = pickle.load(pi, encoding='latin1') 
X = data["data"]
y = data["labels"]
y = tf.keras.utils.to_categorical(y, num_classes=10)

#Normen der Daten
X = X / 255.0
X = X.reshape((10000, 3, 32, 32))

def erstellen():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(3, 32, 32)),
        layers.MaxPooling2D((2, 2), padding='same'),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax') 
    ])


    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def trainieren(modell,X,y):
    modell.fit(X, y, epochs=10)

def speichern(modell):
    modell.save()
