import os
import pickle

import numpy as np
import keras
from src import models
from tensorflow import data as tf_data

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import helpers
from src.helpers import plot_training_history, evaluate_model, predict_and_plot
from src.models import load_model_1, create_test_model, load_test_model

class_names = ["Art Nouveau Modern", "Baroque", "Cubism", "Expressionism", "Impressionism", "Naive Art Primitivism", "Northern Renaissance", "Post Impressionism", "Realism", "Rococo", "Romanticism", "Symbolism"]

'''helpers.sanitize_filenames("../data/processed", delete_invalid=False)'''


#dataprep = helpers.DataPrep()
#dataprep.prepare_data()


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
        shuffle= True,
        class_mode="categorical")


model = load_model_1()

# Kompilieren des Modells
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Modellübersicht
model.summary()

'''model.save("../models/test_model.keras")

tensorboard = keras.callbacks.TensorBoard(log_dir="../logs")
history = model.fit(
    train_generator,
    epochs=2,
    callbacks=[tensorboard],
    validation_data=validation_generator
)

with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

plot_training_history(history, "Test_Modell")
evaluate_model(model, test_generator, "Test_Modell")'''
predict_and_plot(model, test_generator, class_names)

