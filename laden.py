import tensorflow as tf
import pickle 

def laden(batch):
    with open("Daten/data_batch_"+str(batch),"rb") as pi:
        data = pickle.load(pi, encoding='latin1') 
    X = data["data"]
    y = data["labels"]      
    
    X = X / 255.0
    X = X.reshape((10000, 3, 32, 32))
    y = tf.keras.utils.to_categorical(y, num_classes=10)
    return [X,y]

