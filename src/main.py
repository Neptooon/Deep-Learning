import os
import numpy as np
import keras
from keras import layers, Sequential
from keras.src.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow import data as tf_data

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import helpers


'''helpers.sanitize_filenames("../data/processed", delete_invalid=False)'''

'''# Datensatz generieren
image_size = (180, 180)
batch_size = 128

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "../data/processed",
    validation_split=0.1,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Ersten 9 Bilder visualisieren
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.figure()
plt.show()'''

#dataprep = helpers.DataPrep()
#dataprep.prepare_data()

'''data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")
plt.show()'''

train_gen = ImageDataGenerator(
        rescale=1./255.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
                   )
validation_gen =  ImageDataGenerator(
        rescale=1./255.)

test_gen =  ImageDataGenerator(
            rescale=1./255.)

train_generator = train_gen.flow_from_directory(
        '../data/splits/train',
        target_size=(256, 256),
        batch_size=64,
        class_mode="categorical")
validation_generator = validation_gen.flow_from_directory(
        '../data/splits/val',
        target_size=(256, 256),
        batch_size=64,
        class_mode="categorical")

test_generator = test_gen.flow_from_directory(
        '../data/splits/test',
        target_size=(256, 256),
        batch_size=64,
        shuffle= False,
        class_mode="categorical")


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

# Kompilieren des Modells
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Modellübersicht
model.summary()

tensorboard = keras.callbacks.TensorBoard(log_dir="./logs")
history = model.fit(
    train_generator,
    epochs=25,
    callbacks=[tensorboard],
    validation_data=validation_generator
)

model.save("../models/first_model.keras")

loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")
