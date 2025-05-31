import os
import pickle
import numpy as np
import keras
from keras.metrics import TopKCategoricalAccuracy
import keras.utils, keras.optimizers, keras.callbacks
from src import models
from tensorflow import data as tf_data
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import helpers
from src.helpers import plot_training_history, evaluate_model, predict_and_plot, EpochTimer
from src.models import create_resnetlike_model, \
    load_restlike_model, basic_deep_cnn, cnn_dropout_batch, cnn_residual, \
    cnn_with_stride, cnn_with_inception, small_resnet_style, custom_resnet_style_wide_filters, \
    custom_resnet_bottleneck, create_resnetlike_model_new, load_basic_deep_cnn, load_cnn_dropout_batch, \
    load_cnn_residual, load_cnn_with_stride, load_small_resnet, load_cnn_with_inception, load_resnet_style_wide_filters, \
    load_small_resnet_bottleneck, load_resnetlike_model_new
from src.transfer import resnet50, unfreeze

# Klassenbezeichnungen (alphabetisch)
class_names = ["Art Nouveau Modern", "Baroque", "Cubism", "Expressionism", "Impressionism", "Naive Art Primitivism", "Northern Renaissance", "Post Impressionism", "Realism", "Rococo", "Romanticism", "Symbolism"]

# Optional: Umbenennen/Filtern von Dateinamen (auskommentiert)
'''helpers.sanitize_filenames("../data/processed", delete_invalid=False)'''

# GPU-Verfügbarkeit prüfen
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())

# Einmalige Datenaufbereitung
#dataprep = helpers.DataPrep()
#dataprep.prepare_data()

# Trainingsgenerator mit Augmentierung (bereits vorab generiert)
train_gen = ImageDataGenerator(rescale=1./255.)
train_generator = train_gen.flow_from_directory(
    '../data/augmented_fixed',
    target_size=(224, 224),
    batch_size=64,
    shuffle=True,
    class_mode='categorical'
)

# Validierungsdaten
validation_gen = ImageDataGenerator(rescale=1./255.)
validation_generator = validation_gen.flow_from_directory(
    '../data/splits/val',
    target_size=(224, 224),
    batch_size=64,
    shuffle=False,
    class_mode='categorical'
)

# Testdaten
test_gen = ImageDataGenerator(rescale=1./255.)
test_generator = test_gen.flow_from_directory(
    '../data/splits/test',
    target_size=(224, 224),
    batch_size=64,
    shuffle=False,
    class_mode='categorical'
)

# Modell laden
model, basemodel = resnet50()
model_name = 'Resnet50'



# Optional: Lernratenplanung (aktuell deaktiviert)
'''lr_s = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=4e-3,
    decay_steps=10000,
    decay_rate=0.96,
)'''

"""lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)"""

# Kompilieren des Modells
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), # 0,004, 0,003
    loss='categorical_crossentropy',
    metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
)

# Modellübersicht
model.summary()

# Callbacks definieren
tensorboard = keras.callbacks.TensorBoard(log_dir=f"../logs/{model_name}")
timer = EpochTimer(model_name=model_name)
backup_callback = helpers.SaveEveryNEpochs(
    n=5,
    base_path='../models/backup/{}_backup_epoch_{{epoch:02d}}.h5'.format(model_name)
)
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=3, restore_best_weights=True)
#lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)

print(f"Anzahl gesamt Layer: {len(basemodel.layers)}")
# Training des Modells
history = model.fit(
    train_generator,
    epochs=5,
    #callbacks = [tensorboard, backup_callback,timer],
    validation_data=validation_generator,
)

unfreeze(basemodel, num_layers=10)


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), # 0,00001
    loss='categorical_crossentropy',
    metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
)

history_finetune = model.fit(
    train_generator,
    epochs=15,
    #callbacks = [tensorboard, backup_callback,timer],
    validation_data=validation_generator,
)


# Finales Modell speichern
model.save(f"../models/{model_name}.h5")

# Trainingsverlauf als Pickle-Datei speichern
'''with open(f"{model_name}_history.pkl", "wb") as f:
    pickle.dump(history.history, f)'''

# Verlauf laden
'''with open(f"{model_name}_history.pkl", "rb") as f:
    history_data = pickle.load(f)'''

# Visualisierung und Auswertung
plot_training_history(history, f"{model_name}_Modell")
evaluate_model(model, test_generator, f"{model_name}_Modell")

#predict_and_plot(model, test_generator, class_names)
# Visualisierung und Auswertung
helpers.predict_and_plot_with_gradcam_vis(model, test_generator, class_names, last_conv_layer_name="conv2d")

# Abschlussbericht
loss, acc, tkc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.2f}")
print(f"TKC: {tkc*100:.2f}%")


