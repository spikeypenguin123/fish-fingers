from tensorflow.keras.models import load_model
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

class FishClassifier:

    def __init__(self, path_to_weights):
        self.model = load_model(path_to_weights)
        self.classes = ["Aplodactylidae", "Aulopidae", "Carangidae", "Cheilodactylidae", "Dasyatidae", "Diadematidae", "Dinolestidae", "Enoplosidae", "Girellidae", "Heterodontidae", "Kyphosidae", "Labridae", "Lethrinidae", "Microcanthidae", "Monacanthidae", "Monodactylidae", "Plesiopidae", "Pomacentridae", "Scorpaenidae", "Sparidae", "Syngnathidae"]

    def predict(self, image):
        resized = tf.image.resize(image, (256, 256))
        yhat = self.model.predict(np.expand_dims(resized/255, 0))
        top = np.argmax(yhat)
        return (self.classes[top], yhat[0][top])
