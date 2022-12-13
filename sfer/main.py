import numpy

from classifier import Classifier
from loader import PrebuiltModel


def main():
    model_name = PrebuiltModel.EfficientNetB0
    img_size = 64

    classifier = Classifier(model_name, img_size)

    img = numpy.random.rand(1, 64, 64, 1)
    emotion, score = classifier.predict(img)
    print(emotion, score)


if __name__ == "__main__":
    main()
