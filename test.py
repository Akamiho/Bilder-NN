import pickle
import laden
import numpy as np
import matplotlib.pyplot as plt
import show


bilder = laden.train()[0]
bild = bilder[0]



plt.imshow(bild)
plt.show()

show.image(1,0)