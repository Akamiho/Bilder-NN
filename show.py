import pickle 
import matplotlib.pyplot as plt


def image(batch,number):
    with open("Daten/data_batch_"+str(batch),"rb") as pi:
        data = pickle.load(pi, encoding='latin1') 

    bilder = data["data"].reshape((10000, 3, 32, 32))
    bilder = bilder.transpose(0, 2, 3, 1)
    plt.imshow(bilder[number])
    plt.show()



