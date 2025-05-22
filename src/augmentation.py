import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tqdm import tqdm

# Zielverzeichnis (wird neu geschrieben!)
output_dir = "../data/augmented_fixed"
os.makedirs(output_dir, exist_ok=True)

# Anzahl der Augmentierungen pro Originalbild
num_augmented = 2

# Konfiguration des ImageDataGenerators für Augmentierungen
save_aug_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.10,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest',
)

# Pfad zum Original-Trainingsordner
input_dir = "../data/splits/train"

# Iteration über alle Klassen
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Ziel-Ordner erstellen
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    # Augmentierung aller gültigen Bilddateien in der Klasse
    for fname in tqdm(os.listdir(class_path), desc=class_name):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(class_path, fname)
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Augmentierungen generieren und speichern
        i = 0
        for batch in save_aug_gen.flow(x, batch_size=1):
            aug_img = array_to_img(batch[0])
            new_fname = os.path.splitext(fname)[0] + f"_aug{i+1}.jpg"
            aug_img.save(os.path.join(output_class_path, new_fname))
            i += 1
            if i >= num_augmented:
                break # Nur gewünschte Anzahl pro Bild erzeugen
