from keras.metrics import TopKCategoricalAccuracy
import keras.optimizers, keras.callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.helpers import evaluate_model, predict_and_plot_single_image_with_gradcam
from tensorflow.keras.applications.resnet50 import preprocess_input

class_names = ["Art Nouveau Modern", "Baroque", "Cubism", "Expressionism", "Impressionism", "Naive Art Primitivism", "Northern Renaissance", "Post Impressionism", "Realism", "Rococo", "Romanticism", "Symbolism"]

# /content/dataset/test

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_gen.flow_from_directory(
    '/content/dataset/test',
    target_size=(224, 224),
    batch_size=64,
    shuffle=False,
    class_mode='categorical'
)

model_v2 = keras.models.load_model("/content/drive/MyDrive/DeepLearning/models/Transfer_ResNet50_Regularization_V2.h5")
model_v2_name= "ResNet50_V2"

model_v2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
)

evaluate_model(model_v2, test_generator, f"{model_v2_name}_Modell")
predict_and_plot_single_image_with_gradcam(
    model=model_v2,
    image_path="../data/splits/test/Baroque/adriaen-van-de-venne_what-won-t-people-do-for-money.jpg",
    class_names=class_names,
    last_conv_layer_name="conv5_block3_out",
    target_size=(224, 224)  # z. B. je nach Modell
)
loss, acc, tkc = model_v2.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.2f}")
print(f"TKC: {tkc*100:.2f}%")

########################################################################################################################

model_v3 = keras.models.load_model("/content/drive/MyDrive/DeepLearning/models/Transfer_ResNet50_Regularization_V3.h5")
model_v3_name= "ResNet50_V3"

model_v3.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
)

evaluate_model(model_v3, test_generator, f"{model_v3_name}_Modell")
predict_and_plot_single_image_with_gradcam(
    model=model_v3,
    image_path="../data/splits/test/Baroque/adriaen-van-de-venne_what-won-t-people-do-for-money.jpg",
    class_names=class_names,
    last_conv_layer_name="conv5_block3_out",
    target_size=(224, 224)  # z. B. je nach Modell
)
loss, acc, tkc = model_v3.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.2f}")
print(f"TKC: {tkc*100:.2f}%")

########################################################################################################################

model_v4 = keras.models.load_model("/content/drive/MyDrive/DeepLearning/models/backup/Transfer_ResNet50_Regularization_V4_backup_epoch_20.h5")
model_v4_name= "ResNet50_V3"

model_v4.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
)

evaluate_model(model_v4, test_generator, f"{model_v4_name}_Modell")
predict_and_plot_single_image_with_gradcam(
    model=model_v4,
    image_path="../data/splits/test/Art_Nouveau_Modern/akseli-gallen-kallela_the-lair-of-the-lynx-1906.jpg",
    class_names=class_names,
    last_conv_layer_name="conv5_block3_out",
    target_size=(224, 224)  # z. B. je nach Modell
)
loss, acc, tkc = model_v4.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.2f}")
print(f"TKC: {tkc*100:.2f}%")