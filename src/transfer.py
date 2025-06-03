from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50


def resnet50():

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model._name = "resnet50_base"
    base_model.trainable = False


    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = layers.Dense(12, activation='softmax')(x)
    model = keras.Model(inputs, outputs)


    return model


def unfreeze(basemodel):
    for layer in basemodel.layers:
        layer.trainable = False
    for layer in basemodel.layers:
        if "conv5_" in layer.name and not isinstance(layer, keras.layers.BatchNormalization) and not isinstance(layer, keras.layers.Dropout):  # BN eingefroren lassen
            layer.trainable = True

def load_resnet50():
    model = keras.models.load_model("/content/drive/MyDrive/DeepLearning/models/Transfer_ResNet50.h5")
    basemodel = model.get_layer("resnet50_base")
    return model, basemodel