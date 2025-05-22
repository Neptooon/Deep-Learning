import keras
from keras import Sequential, layers
from keras.layers import *
from keras.saving.save import load_model
import tensorflow as tf
from tensorflow.keras import layers, models


# 10 Modelle ------------------------------------------------------------------------------------

# Model 1 BASIC DEEP CNN
def basic_deep_cnn():
    """
    Erstellt ein einfaches Deep Convolutional Neural Network (CNN) zur Bildklassifikation.

    Architektur:
        - 3 Convolution-Blöcke mit zunehmender Filteranzahl (32, 64, 128)
        - MaxPooling nach jedem Block zur Reduktion der Feature-Map-Größe
        - Dense-Schichten für die finale Klassifikation

    Returns:
        keras.Sequential: Kompiliertes Keras-Modell mit Softmax-Ausgabe für 12 Klassen.
    """
    model = keras.Sequential([
        # Convolution Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),

        # Convolution Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),

        # Convolution Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),

        # Klassifikationskopf
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(12, activation='softmax')  # Softmax für 12 Klassen
    ])
    return model

def load_basic_deep_cnn():
    model = keras.models.load_model("../models/BASIC_DEEP_CNN.h5")
    return model


# Model 2 CNN mit Dropout + Batch-norm
def cnn_dropout_batch():
    """
    Erstellt ein CNN mit Batch Normalization und Dropout zur Verbesserung der Trainingsstabilität
    und Reduktion von Overfitting.

    Architektur:
        - Convolutional-Blöcke mit BatchNormalization nach jeder Convolution
        - Dropout-Schicht vor dem letzten Dense-Layer
        - ReLU-Aktivierung in allen versteckten Schichten
        - Softmax-Ausgabe zur Mehrklassenklassifikation (12 Klassen)

    Returns:
        keras.Sequential: Kompiliertes CNN-Modell mit regulierenden Komponenten.
    """
    model = keras.Sequential([
        # Block 1: Convolution + BatchNorm
        layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),

        # Block 2
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),

        # Block 3
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Klassifikationskopf
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),  # Dropout zur Regularisierung
        layers.Dense(12, activation='softmax')
    ])
    return model

def load_cnn_dropout_batch():
    model = keras.models.load_model("../models/CNN_DROPOUT_BATCH.h5")
    return model

# Model 3 CNN Inception
def cnn_with_inception():
    """
    Erstellt ein CNN mit selbstdefinierten Inception-Blöcken zur parallelen Merkmalsextraktion
    mit verschiedenen Filtergrößen.

    Architektur:
        - Initiale Convolution + Pooling
        - Zwei gestapelte Inception-Blöcke mit mehreren Pfaden (1x1, 3x3, 5x5, MaxPool)
        - Global Average Pooling zur Reduktion der Dimensionalität
        - Dense-Schichten für Klassifikation

    Returns:
        keras.Model: Ein funktionales Keras-Modell mit Inception-ähnlicher Architektur.
    """
    def custom_inception_block(x, f1, f2_1, f2_3, f3_1, f3_5, f4):
        """
        Inception-Block mit vier parallelen Pfaden und anschließender Konkatenation.
        """
        path1 = layers.Conv2D(f1, 1, activation='relu', padding='same')(x)

        path2 = layers.Conv2D(f2_1, 1, activation='relu', padding='same')(x)
        path2 = layers.Conv2D(f2_3, 3, activation='relu', padding='same')(path2)

        path3 = layers.Conv2D(f3_1, 1, activation='relu', padding='same')(x)
        path3 = layers.Conv2D(f3_5, 5, activation='relu', padding='same')(path3)

        path4 = layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
        path4 = layers.Conv2D(f4, 5, activation='relu', padding='same')(path4)

        return layers.concatenate([path1, path2, path3, path4])

    inputs = keras.Input(shape=(224, 224, 3))
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = layers.Conv2D(64, 1, activation='relu', padding='same')(x)
    x = layers.Conv2D(192, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = custom_inception_block(x, 64, 96, 128, 16, 32, 32)
    x = custom_inception_block(x, 128, 128, 192, 32, 96, 64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = custom_inception_block(x, 192, 128, 128, 16, 32, 64)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(12, activation='softmax')(x)

    return keras.Model(inputs, outputs)

def load_cnn_with_inception():
    model = keras.models.load_model("../models/CNN_WITH_INCEPTION.h5")
    return model

# Model 4 Convolutions mit stride statt MaxPooling
def cnn_with_stride():
    """
    Erstellt ein CNN, das anstelle von Pooling-Layern Convolutional-Schichten mit Strides nutzt.

    Architektur:
        - Strided Convolutions für Downsampling
        - Batch-Normalisierung zur Stabilisierung
        - Global Average Pooling + Dense-Schichten

    Returns:
        keras.Sequential: Modell mit reduzierter Komplexität durch Strides statt MaxPooling.
    """
    model = keras.Sequential([
        layers.Conv2D(32, 3, strides=3, activation='relu', input_shape=(224, 224, 3)),
        layers.Conv2D(64, 3, strides=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, strides=2, activation='relu'),
        layers.Conv2D(128, 3, strides=2, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, activation='relu'),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(128, activation='relu'),
        layers.Dense(12, activation='softmax')
    ])
    return model

def load_cnn_with_stride():
    model = keras.models.load_model("../models/CNN_WITH_STRIDE.h5")
    return model

# Model 5 Residual CNN
def cnn_residual():
    """
    Erstellt ein einfaches Residual-Netzwerk (ähnlich zu ResNet) mit Skip Connections zur
    Verbesserung des Gradientenflusses in tiefen Netzen.

    Architektur:
        - Convolutional-Blöcke mit Residual-Verbindungen (Add)
        - MaxPooling + GlobalAveragePooling
        - Dense-Schichten für Klassifikation

    Returns:
        keras.Model: Residual CNN mit manuell implementierten Skip Connections.
    """
    inputs = keras.Input(shape=(224, 224, 3))
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    res = x  # Skip-Verbindung

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Add()([x, res])
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(12, activation='softmax')(x)

    return keras.Model(inputs, outputs)

def load_cnn_residual():
    model = keras.models.load_model("../models/CNN_RESIDUAL.h5")
    return model

# Model 6
def small_resnet_style():
    """
    Kleines ResNet-artiges CNN mit eigenen Residual-Blöcken und Batch-Normalisierung.

    Architektur:
        - Residual-Blöcke mit 2x Conv2D + Add
        - Übergänge durch Downsampling
        - Klassifikation durch Dense-Schichten

    Returns:
        keras.Model: ResNet-ähnliches Modell mit geringerer Tiefe.
    """
    def custom_residual_block(x, filters):
        skip_connection = x
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, skip_connection])
        x = layers.Activation('relu')(x)
        return x

    inputs = keras.Input(shape=(224, 224, 3))
    x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)

    x = custom_residual_block(x, 64)
    x = custom_residual_block(x, 64)

    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)

    x = custom_residual_block(x, 128)
    x = custom_residual_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(12, activation='softmax')(x)

    return keras.Model(inputs, outputs)

def load_small_resnet():
    model = keras.models.load_model("../models/CNN_CUSTOM_RESNET_STYLE.h5")
    return model


# Model 7 Wide Filter Resnetlike
def custom_resnet_style_wide_filters():
    """
    Erstellt ein ResNet-artiges CNN mit breiten Convolution-Filtern und angepassten Residual-Blöcken.

    Merkmale:
        - Breite Kernels (z.B. 7x7) in Residual-Blöcken
        - Eignung zur Erfassung großflächiger Merkmale
        - Global Average Pooling + Dense-Schichten

    Returns:
        keras.Model: Modell mit erweiterten Rezeptivfeldern durch breite Filter.
    """
    def custom_residual_block_wide(x, filters, kernel_size=7):
        skip_connection = x
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, skip_connection])
        x = layers.Activation('relu')(x)
        return x

    inputs = keras.Input(shape=(224, 224, 3))

    # Größerer Eingangsfilter
    x = layers.Conv2D(64, 9, strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)

    # Breite Residual Blöcke
    x = custom_residual_block_wide(x, 64, kernel_size=7)
    x = custom_residual_block_wide(x, 64, kernel_size=7)

    # Transition mit breitem Filter
    x = layers.Conv2D(128, 5, strides=2, padding='same', activation='relu')(x)

    x = custom_residual_block_wide(x, 128, kernel_size=7)
    x = custom_residual_block_wide(x, 128, kernel_size=7)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(12, activation='softmax')(x)

    return keras.Model(inputs, outputs)

def load_resnet_style_wide_filters():
    model = keras.models.load_model("../models/CNN_RESNET_STYLE_WIDE_FILTERS.h5")
    return model

# Model 8 Smal Resnetlike Bottleneck
def custom_resnet_bottleneck():
    """
    Erstellt ein ResNet-ähnliches Modell mit Bottleneck-Blöcken zur Reduktion der Rechenkosten.

    Merkmale:
        - 1x1 - 3x3 - 1x1 Architektur in Residual-Blöcken
        - Projektion der Skip-Verbindung bei Dimensionsänderung
        - Dropout zur Regularisierung

    Returns:
        keras.Model: ResNet-Modell mit Bottleneck-Strategie.
    """
    def custom_residual_block_bottleneck(x, filters, stride=1):
        channels = x.shape[-1]
        skip_connection = x

        x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters * 4, kernel_size=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        if channels != filters * 4 or stride != 1:
            # Falls Channel vom Block nicht übereinstimmen
            skip_connection = layers.Conv2D(filters * 4, kernel_size=1, strides=stride, padding='same', use_bias=False)(
                skip_connection)
            skip_connection = layers.BatchNormalization()(skip_connection)

        x = layers.Add()([x, skip_connection])
        x = layers.Activation('relu')(x)

        return x

    inputs = keras.Input(shape=(224, 224, 3))

    x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    x = custom_residual_block_bottleneck(x, 64)
    x = custom_residual_block_bottleneck(x, 64)

    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)

    x = custom_residual_block_bottleneck(x, 128, stride=2)
    x = custom_residual_block_bottleneck(x, 128, stride=2)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(12, activation='softmax')(x) # Output Layer

    return keras.Model(inputs, outputs)

def load_small_resnet_bottleneck():
    model = keras.models.load_model("../models/CNN_RESNET_BOTTLENECK.h5")
    return model

# Model 9
def create_resnetlike_model():
    """
    Erstellt ein großes ResNet-inspiriertes Modell mit skalierter Filteranzahl und 4 Stufen.

    Architektur:
        - Initiale Konvolution + MaxPooling
        - Vier Stufen mit 3, 4, 6, 3 Residual-Blöcken
        - Bottleneck-Block-Architektur (1x1 - 3x3 - 1x1)
        - Global Average Pooling + 2 Dense-Schichten

    Returns:
        keras.Model: Tiefes CNN mit ResNet-Logik zur Bildklassifikation.
    """
    def residual_block(x, filters, stride):
        shortcut = x
        if stride != 1 or x.shape[-1] != filters * 4:
            shortcut = layers.Conv2D(filters * 4, 1, strides=stride, use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Conv2D(filters, 1, strides=1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters * 4, 1, strides=1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    input_shape = (224, 224, 3)
    num_classes = 12
    scale = 1.5  # Skalierung der Filteranzahl

    inputs = layers.Input(shape=input_shape)

    # Eingangskonvolution
    x = layers.Conv2D(int(64 * scale), kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Stage 1: 3 Blöcke (Filters = 64 * scale)
    filters = int(64 * scale)
    for i in range(3):
        x = residual_block(x, filters, stride=1)

    # Stage 2: 4 Blöcke (Filters = 128 * scale)
    filters = int(128 * scale)
    for i in range(4):
        stride = 2 if i == 0 else 1
        x = residual_block(x, filters, stride=stride)

    # Stage 3: 6 Blöcke (Filters = 256 * scale)
    filters = int(256 * scale)
    for i in range(6):
        stride = 2 if i == 0 else 1
        x = residual_block(x, filters, stride=stride)

    # Stage 4: 3 Blöcke (Filters = 512 * scale)
    filters = int(512 * scale)
    for i in range(3):
        stride = 2 if i == 0 else 1
        x = residual_block(x, filters, stride=stride)

    # Output Layer
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation='relu')(x)  # 1. Dense Layer
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256, activation='relu')(x)  # 2. Dense Layer

    outputs = layers.Dense(num_classes, activation='softmax')(x) # Output Layer
    return models.Model(inputs, outputs)


def load_restlike_model():
    model = keras.models.load_model("../models/resnet.h5")
    return model


# Model 10 Big Resnetlike Tuned
def create_resnetlike_model_new():
    """
    Optimierte Version eines großen ResNet-Modells mit angepasster Architektur:

    Merkmale:
        - Optimierter Dropout-Wert (0.5)
        - 4 Stufen (3, 4, 6, 1 Residual-Blöcke)
        - Keine Bottleneck-Struktur, sondern klassische 3x3-ResNet-Architektur
        - Projektion in Shortcut-Verbindung bei Dimensionsunterschieden

    Returns:
        keras.Model: Weiterentwickeltes ResNet-Modell mit guter Generalisierungsfähigkeit.
    """
    def residual_block_new(x, filters, stride):

        shortcut = x

        x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        if shortcut.shape[-1] != filters: # Dimensionen anpassen
            shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut]) # Block Hinzufügen
        x = layers.Activation('relu')(x)
        return x

    input_shape = (224, 224, 3)
    num_classes = 12
    scale = 1  # Skalierung der Filteranzahl

    inputs = layers.Input(shape=input_shape)

    # Eingangskonvolution
    x = layers.Conv2D(int(64 * scale), kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)


    # Stage 1: 3 Blöcke (Filters = 64 * scale)
    filters = int(64 * scale)
    for i in range(3):
        x = residual_block_new(x, filters, stride=1)

    # Stage 2: 4 Blöcke (Filters = 128 * scale)
    filters = int(128 * scale)
    for i in range(4):
        stride = 2 if i == 0 else 1
        x = residual_block_new(x, filters, stride=stride)

    # Stage 3: 6 Blöcke (Filters = 256 * scale)
    filters = int(256 * scale)
    for i in range(6):
        stride = 2 if i == 0 else 1
        x = residual_block_new(x, filters, stride=stride)

    # Stage 4: 1 Block (Filters = 512 * scale)
    filters = int(512 * scale)
    for i in range(1):
        stride = 2 if i == 0 else 1
        x = residual_block_new(x, filters, stride=stride)


    # Output Layer
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation='relu')(x)  # 1. Dense Layer
    x = layers.Dropout(0.5)(x) # Änderung von Dropout 0.25 -> 0.5
    x = layers.Dense(256, activation='relu')(x)  # 2. Dense Layer


    outputs = layers.Dense(num_classes, activation='softmax')(x) # Output Layer
    return models.Model(inputs, outputs)

def load_resnetlike_model_new():
    model = keras.models.load_model("../models/backup/NEW_RESNET_P2_backup_epoch_25.h5")
    return model