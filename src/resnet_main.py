import keras.callbacks
import keras.optimizers
import tensorflow as tf
from keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

import helpers
from transfer import load_resnet50
from transfer import resnet50v1, resnet50v2, resnet50v3, unfreeze


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
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=64,
    shuffle=True,
    class_mode='categorical'
)

# Validierungsdaten
validation_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = validation_gen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=64,
    shuffle=False,
    class_mode='categorical'
)

model_name = 'Transfer_ResNet50_Regularization_V3'

tensorboard = keras.callbacks.TensorBoard(log_dir=f"{LOGS_DIR}/{model_name}")
timer = helpers.EpochTimer(model_name=model_name)
backup_callback = helpers.SaveEveryNEpochs(
    n=5,
    base_path='{}/backup/{}_backup_epoch_{{epoch:02d}}.h5'.format(MODELS_DIR, model_name)
)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5, restore_best_weights=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
)

# Init-Modell laden
# model = resnet50v1 | resnet50v2 | resnet50v3 für das Training des Kopfes

# Modell mit trainiertem Head laden
model = load_resnet50()
base_model = model.get_layer("resnet50_base")
base_model.summary()

# Head Trainng für 10 Epochen
"""model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), # 0,004, 0,003
    loss='categorical_crossentropy',
    metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
)

# Training des Modells
history = model.fit(
    train_generator,
    epochs=10,
    callbacks = [tensorboard, backup_callback,timer, early_stop],
    validation_data=validation_generator,
)"""



#Start: Fine-Tuning mit graduellem Auftauen
unfreeze(base_model)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
)


history_finetune = model.fit(
    train_generator,
    epochs=30,
    initial_epoch=20,
    callbacks = [tensorboard, backup_callback,timer, early_stop], #lr_scheduler
    validation_data=validation_generator,
)


# Finales Modell speichern
model.save(f"{MODELS_DIR}/{model_name}.h5")
model.save(f"{MODELS_DIR}/{model_name}.keras")
