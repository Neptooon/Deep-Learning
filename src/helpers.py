import os
import unicodedata
import re
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random


def sanitize_filenames(base_path, delete_invalid=False):
    """
    Bereinigt Dateinamen in einem Verzeichnisbaum von ungültigen oder nicht-ASCII-Zeichen.

    Args:
        base_path (str): Pfad zum Basisverzeichnis.
        delete_invalid (bool): Wenn True, löscht Dateien, die nicht reparierbar sind.
    """
    for root, dirs, files in os.walk(base_path):
        for fname in files:
            original_path = os.path.join(root, fname)

            # Versuche, den Dateinamen als UTF-8 zu interpretieren
            try:
                fname.encode("utf-8").decode("utf-8")
            except UnicodeDecodeError:
                print(f"Ungültige Kodierung: {original_path}")

            # Entferne alle Zeichen, die nicht ASCII oder ungültig im Dateinamen sind
            clean_name = unicodedata.normalize('NFKD', fname)
            clean_name = clean_name.encode('ascii', 'ignore').decode('ascii')
            clean_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', clean_name)
            clean_name = clean_name.strip()

            if not clean_name:
                if delete_invalid:
                    print(f"Lösche ungültige Datei: {original_path}")
                    os.remove(original_path)
                continue

            new_path = os.path.join(root, clean_name)
            if new_path != original_path:
                print(f"Umbenennen: {original_path} -> {new_path}")
                os.rename(original_path, new_path)


class DataPrep:
    def __init__(self, raw_data_dir='../data/processed/', target_dir='../data/splits/', max_files=2000, random_state=42):
        self.raw_data_dir = raw_data_dir
        self.target_dir = target_dir
        self.max_files = max_files
        self.random_state = random_state

    def create_directories(self):
        os.makedirs(self.target_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            for class_name in os.listdir(self.raw_data_dir):
                os.makedirs(os.path.join(self.target_dir, split, class_name), exist_ok=True)

    def prepare_data(self):
        self.create_directories()

        class_names = os.listdir(self.raw_data_dir)
        for class_name in tqdm(class_names, desc="Verarbeitung der Klassen"):
            class_dir = os.path.join(self.raw_data_dir, class_name)
            images = os.listdir(class_dir)
            random.shuffle(images)

            selected_images = images[:self.max_files]

            train_images, remaining_images = train_test_split(selected_images, test_size=0.3, random_state=self.random_state)
            val_images, test_images = train_test_split(remaining_images, test_size=0.66, random_state=self.random_state)

            self.copy_images(train_images, class_dir, class_name, "train")
            self.copy_images(val_images, class_dir, class_name, "val")
            self.copy_images(test_images, class_dir, class_name, "test")

    def copy_images(self, image_list, class_dir, class_name, split):
        for image in tqdm(image_list, desc=f"Kopiere {split} Bilder für {class_name}", leave=False):
            source = os.path.join(class_dir, image)
            destination = os.path.join(self.target_dir, split, class_name, image)
            shutil.copy(source, destination)