import re
from abc import ABC, abstractmethod
from typing import Type, Any, Set

import keras
from keras.engine import InputLayer
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Convolution2D, BatchNormalization, concatenate, Dropout, \
    Input, GlobalAveragePooling2D


class AbstractNetworkFactory(ABC):
    """Abstract base class for neural network factory"""

    @property
    @abstractmethod
    def pattern(self) -> str:
        raise NotImplementedError

    @classmethod
    def matches(cls, name: str) -> bool:
        pattern: Any = cls.pattern
        return bool(re.fullmatch(pattern=pattern, string=name, flags=re.IGNORECASE))

    @staticmethod
    @abstractmethod
    def build(name: str, shape: tuple, classes: int) -> keras.Model:
        pass


class VGG13Factory(AbstractNetworkFactory):
    pattern = "vgg13(like)?"

    @staticmethod
    def build(name: str, shape: tuple, classes: int) -> keras.Model:
        if name.lower() == "vgg13":
            return VGG13Factory.build_original(shape=shape, classes=classes)
        elif name.lower() == "vgg13like":
            return VGG13Factory.build_ferplus(shape=shape, classes=classes)

    @staticmethod
    def build_original(shape: tuple, classes: int) -> keras.Model:
        model = keras.models.Sequential()
        model.add(InputLayer(input_shape=shape))
        for fcount in [64, 128, 256, 512, 512]:
            model.add(Conv2D(filters=fcount, kernel_size=(3, 3), padding='same', activation="relu"))
            model.add(Conv2D(filters=fcount, kernel_size=(3, 3), padding='same', activation="relu"))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=classes, activation="softmax"))

        # assert model.count_params() == 133_047_848
        return model

    @staticmethod
    def build_ferplus(shape: tuple, classes: int) -> keras.Model:
        model = keras.models.Sequential()
        model.add(InputLayer(input_shape=shape))
        for fcount in [64, 128]:
            model.add(Conv2D(filters=fcount, kernel_size=(3, 3), padding='same', activation="relu"))
            model.add(Conv2D(filters=fcount, kernel_size=(3, 3), padding='same', activation="relu"))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(rate=0.25))

        for fcount in [256, 256]:
            model.add(Conv2D(filters=fcount, kernel_size=(3, 3), padding='same', activation="relu"))
            model.add(Conv2D(filters=fcount, kernel_size=(3, 3), padding='same', activation="relu"))
            model.add(Conv2D(filters=fcount, kernel_size=(3, 3), padding='same', activation="relu"))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(rate=0.25))

        model.add(Flatten())
        model.add(Dense(units=1024, activation="relu"))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=1024, activation="relu"))
        model.add(Dropout(rate=0.25))
        model.add(Dense(units=classes, activation="softmax"))

        return model


class VGGFactory(AbstractNetworkFactory):
    """
    Factory-class for VGG
    Paper: Very Deep Convolutional Networks for Large-Scale Image Recognition
    Author: Karen Simonyan, Andrew Zisserman
    Link: https://arxiv.org/pdf/1409.1556.pdf
    """

    pattern = "vgg1(6|9)"

    @staticmethod
    def build(name: str, shape: tuple, classes: int) -> keras.Model:
        constructor = {
            "vgg16": keras.applications.VGG16,
            "vgg19": keras.applications.VGG19,
        }.get(name.lower())
        return constructor(input_shape=shape, classes=classes, include_top=True, weights=None)


class LeNetFactory(AbstractNetworkFactory):
    """
    Factory-class for LeNet
    Paper: Gradient-based learning applied to document recognition
    Author: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
    Link: https://axon.cs.byu.edu/~martinez/classes/678/Papers/Convolution_nets.pdf
    """

    pattern = "lenet(-5)?"

    @staticmethod
    def build(name: str, shape: tuple, classes: int) -> keras.Model:
        # initialize the model
        model = keras.Sequential()
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=shape, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))
        # softmax classifier
        model.add(Dense(classes, activation="softmax"))
        # return the constructed network architecture
        return model


class OwnNet(AbstractNetworkFactory):
    pattern = "(own|custom|my)(net)?"

    @staticmethod
    def build(name: str, shape: tuple, classes: int) -> keras.Model:
        model = keras.Sequential()
        model.add(InputLayer(input_shape=shape))
        for fcount in [64, 128, 256, 512]:
            model.add(Conv2D(fcount, kernel_size=3, activation='relu', padding='same'))
            model.add(Conv2D(fcount, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.25))
        model.add(Flatten())

        for _ in range(2):
            model.add(Dense(units=1024, activation='relu'))
            model.add(Dropout(0.5))

        model.add(Dense(classes, activation='softmax'))
        return model


class SEResNetFactory(AbstractNetworkFactory):
    """
    Factory-class for SEResNet
    Paper: Squeeze-and-Excitation Networks
    Author: Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
    Link: https://arxiv.org/pdf/1709.01507.pdf
    """

    pattern = "seresnet(18|34|50|101|152)"

    @classmethod
    def build(cls, name: str, shape: tuple, classes: int):
        model_fn = cls._get_constructor(name)
        return model_fn(input_shape=shape, classes=classes, include_top=True, weights=None)

    @classmethod
    def _get_constructor(cls, name) -> callable:
        from classification_models.keras import Classifiers
        model_fn, preprocess_input = Classifiers.get(name.lower())
        return model_fn


class ResNeXtFactory(AbstractNetworkFactory):
    """
    Factory-class for ResNeXt
    Paper: Aggregated Residual Transformations for Deep Neural Networks
    Author: Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
    Link: https://arxiv.org/pdf/1611.05431v2.pdf
    """

    pattern = "resnext(50|101)"

    @classmethod
    def build(cls, name: str, shape: tuple, classes: int):
        model_fn = cls._get_constructor(name)
        return model_fn(input_shape=shape, classes=classes, include_top=True, weights=None)

    @classmethod
    def _get_constructor(cls, name) -> callable:
        from classification_models.keras import Classifiers
        model_fn, preprocess_input = Classifiers.get(name.lower())
        return model_fn


class DenseNetFactory(AbstractNetworkFactory):
    """
    Factory-class for DenseNet
    Paper: Densely Connected Convolutional Networks
    Author: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    Link: https://arxiv.org/pdf/1608.06993v5.pdf
    """

    pattern = "densenet(121|169|201)"

    @staticmethod
    def build(name: str, shape: tuple, classes: int) -> keras.Model:
        constructor = {
            "densenet121": keras.applications.DenseNet121,
            "densenet169": keras.applications.DenseNet169,
            "densenet201": keras.applications.DenseNet201,
        }.get(name.lower())
        return constructor(input_shape=shape, classes=classes, include_top=True, weights=None)


class MobileNetFactory(AbstractNetworkFactory):
    """
    Factory-class for MobileNet
    Paper: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    Author:
        Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko,
        Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
    Link: https://arxiv.org/pdf/1704.04861
    """

    pattern = "mobilenet(v2)?"

    @staticmethod
    def build(name: str, shape: tuple, classes: int) -> keras.Model:
        constructor = {
            "mobilenet": keras.applications.MobileNet,
            "mobilenetv2": keras.applications.MobileNetV2,
        }.get(name.lower())
        return constructor(input_shape=shape, classes=classes, include_top=True, weights=None)


class NASNetFactory(AbstractNetworkFactory):
    """
    Factory-class for NASNet
    Paper: Learning Transferable Architectures for Scalable Image Recognition
    Author: Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le
    Link: https://arxiv.org/pdf/1707.07012.pdf
    """

    pattern = "nasnet(large|mobile)"

    @staticmethod
    def build(name: str, shape: tuple, classes: int) -> keras.Model:
        constructor = {
            "nasnetmobile": keras.applications.NASNetMobile,
            "nasnetlarge": keras.applications.NASNetLarge,
        }.get(name.lower())
        return constructor(input_shape=shape, classes=classes, include_top=True, weights=None)


class ResNetFactory(AbstractNetworkFactory):
    """
    Factory-class for ResNet
    Paper: Deep Residual Learning for Image Recognition
    Author: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Link: https://arxiv.org/pdf/1512.03385.pdf
    """

    pattern = "resnet(18|34|50|101|152|200)"

    @classmethod
    def build(cls, name: str, shape: tuple, classes: int):
        import keras_resnet.models
        input_layer = keras.layers.Input(shape)
        constructor = {
            "resnet18": keras_resnet.models.ResNet18,
            "resnet34": keras_resnet.models.ResNet34,
            "resnet50": keras_resnet.models.ResNet50,
            "resnet101": keras_resnet.models.ResNet101,
            "resnet152": keras_resnet.models.ResNet152,
            "resnet200": keras_resnet.models.ResNet200,
        }.get(name.lower())
        base_model: keras.Model = constructor(inputs=input_layer, include_top=False, classes=classes)

        x = base_model.layers[-1].output
        x = GlobalAveragePooling2D()(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dropout(0.5)(x)
        x = Dense(classes, activation='softmax')(x)
        model = keras.Model(inputs=base_model.input, outputs=x)
        return model


class DeXpressionFactory(AbstractNetworkFactory):
    """
    Factory-class for DeXpression
    Paper: DeXpression: Deep Convolutional Neural Network for Expression Recognition
    Author: Peter Burkert, Felix Trier, Muhammad Zeshan Afzal, Andreas Dengel, Marcus Liwicki
    Link: https://arxiv.org/pdf/1509.05371.pdf
    """

    pattern = "dexpression"

    @staticmethod
    def build(name: str, shape: tuple, classes: int) -> keras.Model:
        # START MODEL
        img_input = keras.Input(shape=shape)
        conv_1 = Convolution2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv_1')(img_input)
        maxpool_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv_1)
        x = BatchNormalization()(maxpool_1)

        # FEAT-EX1
        conv_2a = Convolution2D(96, (1, 1), strides=(1, 1), padding='same', activation='relu', name='conv_2a')(x)
        conv_2b = Convolution2D(208, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_2b')(conv_2a)
        maxpool_2a = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='maxpool_2a')(x)
        conv_2c = Convolution2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_2c')(maxpool_2a)
        concat_1 = concatenate([conv_2b, conv_2c], axis=3, name='concat_2')
        maxpool_2b = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool_2b')(concat_1)

        # FEAT-EX2
        conv_3a = Convolution2D(96, (1, 1), strides=(1, 1), padding='same', activation='relu', name='conv_3a')(
            maxpool_2b)
        conv_3b = Convolution2D(208, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_3b')(conv_3a)
        maxpool_3a = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='maxpool_3a')(maxpool_2b)
        conv_3c = Convolution2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_3c')(maxpool_3a)
        concat_3 = concatenate([conv_3b, conv_3c], axis=3, name='concat_3')
        maxpool_3b = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool_3b')(concat_3)

        # FINAL LAYERS
        net = Flatten()(maxpool_3b)
        net = Dense(classes, activation='softmax', name='predictions')(net)

        # Create model.
        model = keras.Model(img_input, net, name='deXpression')
        return model


class InceptionFactory(AbstractNetworkFactory):
    """
    Factory-class for Inception
    Paper: Network In Network
    Author: Min Lin, Qiang Chen, Shuicheng Yan
    Link: https://arxiv.org/pdf/1312.4400v3.pdf
    """

    pattern = "inception(v3|resnetv2)|xception"

    @staticmethod
    def build(name: str, shape: tuple, classes: int) -> keras.Model:
        constructor = {
            "inceptionv3": keras.applications.InceptionV3,
            "inceptionresnetv2": keras.applications.InceptionResNetV2,
            "xception": keras.applications.Xception,
        }.get(name.lower())
        return constructor(input_shape=shape, classes=classes, include_top=True, weights=None)


class EfficientNetFactory(AbstractNetworkFactory):
    """
    Factory-class for EfficientNet
    Paper: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
    Author: Mingxing Tan, Quoc V. Le
    Link: https://arxiv.org/pdf/1905.11946.pdf
    """

    pattern = "efficientnetb[0-7]"

    @classmethod
    def build(cls, name: str, shape: tuple, classes: int):
        # Load pretrained resnet
        model_fn = cls.__get_network(name)
        base_model = model_fn(input_shape=shape, classes=classes, include_top=True, weights=None)
        return base_model

    @classmethod
    def __get_network(cls, name) -> callable:
        import efficientnet.keras as en
        return {
            "efficientnetb0": en.EfficientNetB0,
            "efficientnetb1": en.EfficientNetB1,
            "efficientnetb2": en.EfficientNetB2,
            "efficientnetb3": en.EfficientNetB3,
            "efficientnetb4": en.EfficientNetB4,
            "efficientnetb5": en.EfficientNetB5,
            "efficientnetb6": en.EfficientNetB6,
            "efficientnetb7": en.EfficientNetB7
        }.get(name.lower())


class NetworkFactory:

    @staticmethod
    def get(name: str) -> Type[AbstractNetworkFactory]:
        factories: Set[Type[AbstractNetworkFactory]] = {
            DenseNetFactory,
            DeXpressionFactory,
            EfficientNetFactory,
            InceptionFactory,
            LeNetFactory,
            MobileNetFactory,
            NASNetFactory,
            OwnNet,
            ResNetFactory,
            ResNeXtFactory,
            SEResNetFactory,
            VGGFactory,
            VGG13Factory,
        }

        return next(factory for factory in factories if factory.matches(name))

    @staticmethod
    def load(filepath: str) -> keras.Model:
        """
        Load model from HDF5 file
        Args:
            filepath (str): Path to file containing model data
        Returns:
            keras.Model: Loaded model
        """
        return keras.models.load_model(filepath)
