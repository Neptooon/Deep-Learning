import keras.callbacks
import keras.optimizers
import tensorflow as tf
from keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import helpers
from transfer import load_resnet50
from transfer import resnet50, unfreeze

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

TRAIN_DIR = "/content/dataset/augmented_fixed"
VAL_DIR = "/content/dataset/val"
LOGS_DIR = "/content/drive/MyDrive/DeepLearning/logs"
MODELS_DIR = "/content/drive/MyDrive/DeepLearning/models"


# Trainingsgenerator mit Augmentierung (bereits vorab generiert)
train_gen = ImageDataGenerator(rescale=1./255.)
train_generator = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=64,
    shuffle=True,
    class_mode='categorical'
)

# Validierungsdaten
validation_gen = ImageDataGenerator(rescale=1./255.)
validation_generator = validation_gen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=64,
    shuffle=False,
    class_mode='categorical'
)

# Testdaten
"""test_gen = ImageDataGenerator(rescale=1./255.)
test_generator = test_gen.flow_from_directory(
    '../data/splits/test',
    target_size=(224, 224),
    batch_size=64,
    shuffle=False,
    class_mode='categorical'
)"""




model_name = 'Transfer_ResNet50'

tensorboard = keras.callbacks.TensorBoard(log_dir=f"{LOGS_DIR}/{model_name}")
timer = helpers.EpochTimer(model_name=model_name)
backup_callback = helpers.SaveEveryNEpochs(
    n=5,
    base_path='{}/backup/{}_backup_epoch_{{epoch:02d}}.h5'.format(MODELS_DIR, model_name)
)
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=3, restore_best_weights=True)
#lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)


# Init-Modell laden
model = resnet50()
base_model = model.get_layer("resnet50_base")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), # 0,004, 0,003
    loss='categorical_crossentropy',
    metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
)

# Training des Modells
history = model.fit(
    train_generator,
    epochs=15,
    callbacks = [tensorboard, backup_callback,timer],
    validation_data=validation_generator,
)

unfreeze(base_model)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), # 0,00001
    loss='categorical_crossentropy',
    metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
)


history_finetune = model.fit(
    train_generator,
    epochs=30,
    initial_epoch=15,
    callbacks = [tensorboard, backup_callback,timer],
    validation_data=validation_generator,
)


# Finales Modell speichern
model.save(f"{MODELS_DIR}/{model_name}.h5")

# Trainingsverlauf als Pickle-Datei speichern
"""with open(f"{model_name}_history.pkl", "wb") as f:
    pickle.dump(history.history, f)"""

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#Testen wieder lokal

# Verlauf laden
'''with open(f"{model_name}_history.pkl", "rb") as f:
    history_data = pickle.load(f)'''

"""# Visualisierung und Auswertung
plot_training_history(history, f"{model_name}_Modell")
evaluate_model(model, test_generator, f"{model_name}_Modell")

#predict_and_plot(model, test_generator, class_names)
# Visualisierung und Auswertung
helpers.predict_and_plot_with_gradcam_vis(model, test_generator, class_names, last_conv_layer_name="conv2d")

# Abschlussbericht
loss, acc, tkc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.2f}")
print(f"TKC: {tkc*100:.2f}%")"""