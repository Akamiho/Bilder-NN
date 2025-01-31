import matplotlib.pyplot as plt


def plotLoss(history):
    plt.plot(history.history['val_loss'])
    plt.title('Loss history')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.show()
    
def plotAccuracy(history):
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy history')
    plt.ylabel('Accuracy value (%)')
    plt.xlabel('No. epoch')
    plt.show()