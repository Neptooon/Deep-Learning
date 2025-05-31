from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50


def resnet50():

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False


    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = layers.Dense(12, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model, base_model


def unfreeze(basemodel, num_layers):
    for layer in basemodel.layers[-num_layers:]:
        layer.trainable = True
