import numpy

import network
from loader import download_model, PrebuiltModel, download_emotions


class Classifier(object):
    def __init__(self, model_name: PrebuiltModel, img_size: int):
        filename = download_model(model_name, img_size)
        factory = network.NetworkFactory.get(model_name.value.lower())
        self.model = factory.build(model_name.value.lower(), (64, 64, 1), 7)
        self.model.load_weights(filename)
        self.model.summary()

        emotions_file = download_emotions()
        self.emotions = numpy.load(emotions_file, allow_pickle=True)

    def predict(self, image):
        prediction = self.model.predict(image)
        argmax = numpy.argmax(prediction)
        return self.emotions[argmax], prediction[0][argmax]
