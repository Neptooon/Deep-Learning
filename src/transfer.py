from keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50




def resnet50v1():
    """
    Erstellt ein transferbasiertes ResNet50-Modell (v1) mit:
        - eingefrorener ResNet50-Basis (ImageNet-Gewichte)
        - GlobalAveragePooling
        - BatchNormalization, Dropout
        - 1 Dense-Ausgabeschicht mit Softmax (12 Klassen)

    Returns:
        keras.Model: Kompiliertes ResNet50-Modell mit einfacher Top-Architektur.
    """
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    resnet_base.trainable = False
    base_model = keras.Model(inputs=resnet_base.input, outputs=resnet_base.output, name="resnet50_base")


    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Activation('relu')(x)
    outputs = layers.Dense(12, activation='softmax', kernel_regularizer=regularizers.l2(0.0001))(x) # 0.001
    model = keras.Model(inputs, outputs)

    return model


def resnet50v2():
    """
    ResNet50-Modell (v2) mit zusätzlicher Dense-Schicht vor der Ausgabe.

    Architektur:
        - ResNet50-Basis eingefroren
        - GlobalAveragePooling
        - Dense-Schicht (128 Neuronen) mit L2-Regularisierung
        - BatchNormalization + ReLU + Dropout
        - Softmax-Ausgabe für 12 Klassen

    Returns:
        keras.Model: Transfer-Learning-Modell mit erweiterter Kopfarchitektur.
    """
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    resnet_base.trainable = False
    base_model = keras.Model(inputs=resnet_base.input, outputs=resnet_base.output, name="resnet50_base")


    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    outputs = layers.Dense(12, activation='softmax', kernel_regularizer=regularizers.l2(0.0001))(x) # 0.001
    model = keras.Model(inputs, outputs)

    return model


def resnet50v3():
    """
    Erweiterte Version des ResNet50-Modells (v3) mit stärkeren Regularisierungsmechanismen.

    Unterschiede:
        - Dense-Schicht mit 256 Neuronen
        - Zwei Dropout-Schichten (0.3 und 0.5)
        - L2-Regularisierung für beide Dense-Schichten

    Returns:
        keras.Model: Fortgeschrittene Version für robustes Transfer-Learning.
    """
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    resnet_base.trainable = False
    base_model = keras.Model(inputs=resnet_base.input, outputs=resnet_base.output, name="resnet50_base")


    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(12, activation='softmax', kernel_regularizer=regularizers.l2(0.0001))(x) # 0.001
    model = keras.Model(inputs, outputs)

    return model



def unfreeze(basemodel):
    """
    Gibt gezielt tiefere Convolution-Layer (Block 4 & 5) in einem ResNet-Modell frei
    für weiteres Fine-Tuning.

    BatchNormalization- und Dropout-Schichten bleiben eingefroren, um Training zu stabilisieren.

    Args:
        basemodel (keras.Model): Ein ResNet50-Modell mit eingefrorener Basis.
    """
    layers_unfrozen = 0
    for layer in basemodel.layers:
        layer.trainable = False
    for layer in basemodel.layers:
        if ("conv5_" in layer.name or "conv4_" in layer.name) and not isinstance(layer, keras.layers.BatchNormalization) and not isinstance(layer, keras.layers.Dropout):  # BN eingefroren lassen
            layer.trainable = True
            layers_unfrozen += 1
    print(f"Anzahl unfreezed Layer: {layers_unfrozen}")


def load_resnet50():
    path = "/content/drive/MyDrive/DeepLearning/models/Transfer_ResNet50_Regularization_V4.h5"
    model = keras.models.load_model(path)
    print("Lade Modell aus: ",path)
    return model