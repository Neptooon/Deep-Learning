import os
import pickle

import numpy as np
import keras
from keras.metrics import TopKCategoricalAccuracy

from src import models
from tensorflow import data as tf_data
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import helpers
from src.helpers import plot_training_history, evaluate_model, predict_and_plot, EpochTimer
from src.models import load_model_1, create_test_model, load_test_model, create_model_1

class_names = ["Art Nouveau Modern", "Baroque", "Cubism", "Expressionism", "Impressionism", "Naive Art Primitivism", "Northern Renaissance", "Post Impressionism", "Realism", "Rococo", "Romanticism", "Symbolism"]

'''helpers.sanitize_filenames("../data/processed", delete_invalid=False)'''

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())

#dataprep = helpers.DataPrep()
#dataprep.prepare_data()


train_gen = ImageDataGenerator(rescale=1./255.)
train_generator = train_gen.flow_from_directory(
    '../data/augmented_fixed',
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    class_mode='categorical'
)

validation_gen = ImageDataGenerator(rescale=1./255.)
validation_generator = validation_gen.flow_from_directory(
    '../data/splits/val',
    target_size=(224, 224),
    batch_size=64,
    shuffle=False,
    class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1./255.)
test_generator = test_gen.flow_from_directory(
    '../data/splits/test',
    target_size=(224, 224),
    batch_size=64,
    shuffle=False,
    class_mode='categorical'
)

model = load_model_1()

# Kompilieren des Modells
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'categorical_accuracy', TopKCategoricalAccuracy(k=3)]
)

# Modellübersicht
model.summary()

'''tensorboard = keras.callbacks.TensorBoard(log_dir="../logs")
timer = EpochTimer()
backup_callback = helpers.SaveEveryNEpochs(
    n=5,
    base_path='../models/backup/backup_epoch_{epoch:02d}.h5'
)'''

'''history = model.fit(
    train_generator,
    epochs=25,
    callbacks = [tensorboard, backup_callback, timer],
    validation_data=validation_generator
)'''

#model.save("../models/model_one.h5")

'''with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)'''

#plot_training_history(history, "Test_Modell")
evaluate_model(model, test_generator, "Test_Modell")
predict_and_plot(model, test_generator, class_names)
loss, acc, _, _ = model.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")


