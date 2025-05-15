from keras import layers, Sequential
from keras.src.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.saving import load_model


def create_model_1():
    model = Sequential([
        # 1. Convolutional Layer
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 2. Convolutional Layer
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 3. Convolutional Layer
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 4. Convolutional Layer
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 5. Convolutional Layer
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Übergang zu Fully Connected Layers
        Flatten(),

        # 1. Fully Connected Layer
        Dense(512, activation='relu'),

        # 2. Fully Connected Layer
        Dense(128, activation='relu'),

        # Output Layer (12 Klassen)
        Dense(12, activation='softmax')
    ])
    return model

def load_model_1():
    model = load_model("../models/first_model.keras")
    return model

def create_test_model():
    model = Sequential([
        # 1. Convolutional Layer
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Übergang zu Fully Connected Layers
        Flatten(),

        # 1. Fully Connected Layer
        Dense(512, activation='relu'),

        # 2. Fully Connected Layer
        Dense(128, activation='relu'),

        # Output Layer (12 Klassen)
        Dense(12, activation='softmax')
    ])
    return model

def load_test_model():
    model = load_model("../models/test_model.keras")
    return model

