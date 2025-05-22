import os
from math import ceil
from tf_keras_vis.gradcam import Gradcam
import numpy as np
import unicodedata
import re
import shutil
import tensorflow as tf
import time
import csv
from keras.callbacks import ModelCheckpoint
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2


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
    """
    Klasse zur Vorbereitung und Aufteilung von Bilddaten in Trainings-, Validierungs- und Testsets.

    Attributes:
        raw_data_dir (str): Pfad zum Ursprungsverzeichnis der Daten.
        target_dir (str): Zielverzeichnis für die gesplitteten Daten.
        max_files (int): Maximalanzahl an Bildern pro Klasse.
        random_state (int): Zufallszustand für reproduzierbare Aufteilungen.
    """
    def __init__(self, raw_data_dir='../data/processed/', target_dir='../data/splits/', max_files=2000, random_state=42):
        self.raw_data_dir = raw_data_dir
        self.target_dir = target_dir
        self.max_files = max_files
        self.random_state = random_state

    def create_directories(self):
        """
        Erstellt die Verzeichnisstruktur für Trainings-, Validierungs- und Testdaten
        inklusive Klassenunterordnern.
        """
        os.makedirs(self.target_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            for class_name in os.listdir(self.raw_data_dir):
                os.makedirs(os.path.join(self.target_dir, split, class_name), exist_ok=True)

    def prepare_data(self):
        """
        Führt den kompletten Datenvorbereitungsschritt aus:
            - Erstellt Zielstruktur
            - Wählt zufällig Bilder aus
            - Splittet in train/val/test
            - Kopiert die Dateien in die jeweiligen Zielverzeichnisse
        """
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
        """
        Kopiert Bilder aus einem Klassenverzeichnis in das entsprechende Split-Zielverzeichnis.

        Args:
            image_list (List[str]): Liste von Bilddateinamen.
            class_dir (str): Pfad zum Quell-Klassenordner.
            class_name (str): Name der Klasse.
            split (str): Split-Art ('train', 'val', 'test').
        """
        for image in tqdm(image_list, desc=f"Kopiere {split} Bilder für {class_name}", leave=False):
            source = os.path.join(class_dir, image)
            destination = os.path.join(self.target_dir, split, class_name, image)
            shutil.copy(source, destination)


def plot_training_history(history, model_name):
    """
    Plottet Trainings- und Validierungs-Accuracy sowie -Loss und speichert das Diagramm.

    Args:
        history (History): Keras History-Objekt vom Training.
        model_name (str): Name des Modells zur Dateibenennung.
    """
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'../plots/{model_name}_training_plot.png')  # Abspeichern für Bericht
    plt.show()

def evaluate_model(model, test_generator, model_name):
    """
    Führt eine Evaluation eines Modells mit einem Testgenerator durch und zeigt:
        - Classification Report
        - Normalisierte Confusion Matrix

    Args:
        model (tf.keras.Model): Das zu evaluierende Modell.
        test_generator (DirectoryIterator): Test-Daten-Generator.
        model_name (str): Name des Modells für Titel/Speicherung.
    """
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_generator.classes

    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_prob = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(14, 14))
    sns.heatmap(cm_prob, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=test_generator.class_indices.keys(),
                yticklabels=test_generator.class_indices.keys())
    plt.title(f'Confusion Matrix Probabilities - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'../plots/{model_name}_confusion_matrix.png')
    plt.tight_layout()
    plt.show()

def predict_and_plot(model, generator, class_names, num_images=15):
    """
    Zeigt eine Auswahl an Vorhersagen und deren Ground Truth als Bildübersicht.

    Args:
        model (tf.keras.Model): Das trainierte Modell.
        generator (DirectoryIterator): Daten-Generator mit Bildern und Labels.
        class_names (List[str]): Liste aller Klassennamen.
        num_images (int): Anzahl der anzuzeigenden Beispiele (Standard: 15).
    """
    images, labels = next(generator)
    predictions = model.predict(images)

    loss, acc, topKCategoricalAccuracy = model.evaluate(images[:15], labels[:15], verbose=0)
    print(f"Subset Accuracy: {acc * 100:.2f}%, Loss: {loss:.4f}")

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        ax = plt.subplot(ceil(num_images/5), 5, i + 1)
        plt.imshow(images[i])
        true_label = class_names[np.argmax(labels[i])]
        predicted_label = class_names[np.argmax(predictions[i])]
        plt.title(f"T: {true_label}\nP: {predicted_label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


class SaveEveryNEpochs(tf.keras.callbacks.Callback):
    """
    Callback zum periodischen Speichern des Modells nach jeder n-ten Epoche.

    Attributes:
        n (int): Intervall der Epochen.
        base_path (str): Formatierter Dateipfad mit Platzhalter '{epoch}'.
    """
    def __init__(self, n, base_path):
        super().__init__()
        self.n = n
        self.base_path = base_path

    def on_epoch_end(self, epoch, logs=None):
        """
        Speichert das Modell nach jeder n-ten Epoche.
        """
        if (epoch + 1) % self.n == 0:
            path = self.base_path.format(epoch=epoch + 1)
            self.model.save(path)
            print(f"✅ Modell nach Epoche {epoch + 1} gespeichert: {path}")

def cutout(image):
    """
    Fügt einem Bild zufällig ein schwarzes Rechteck hinzu (Cutout-Augmentation).

    Args:
        image (tf.Tensor): Eingabebild mit Shape (H, W, C).

    Returns:
        tf.Tensor: Bild mit Cutout-Maske.
    """
    h, w, _ = image.shape
    mask_height = tf.random.uniform([], 20, 60, dtype=tf.int32)
    mask_width = tf.random.uniform([], 20, 60, dtype=tf.int32)

    top = tf.random.uniform([], 0, h - mask_height, dtype=tf.int32)
    left = tf.random.uniform([], 0, w - mask_width, dtype=tf.int32)

    # Schwarze Maske erzeugen
    cutout_area = tf.ones((mask_height, mask_width, 3), dtype=image.dtype) * 0.0
    paddings = [[top, h - top - mask_height], [left, w - left - mask_width], [0, 0]]
    mask = tf.pad(cutout_area, paddings, constant_values=1.0)

    return image * mask

class EpochTimer(tf.keras.callbacks.Callback):
    """
    Callback zum Messen und Protokollieren der Trainingszeit pro Epoche.

    Attributes:
        model_name (str): Name des Modells (nicht verwendet).
        log_file (str): CSV-Datei zur Speicherung der Laufzeiten.
    """
    def __init__(self, model_name, log_file='epoch_times.csv'):
        super().__init__()
        self.model_name = model_name
        self.epoch_times = []
        self.log_file = log_file
        self.header_written = False

    def on_epoch_begin(self, epoch, logs=None):
        """Startet Zeitmessung zu Beginn der Epoche."""
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """Speichert Zeitdauer in CSV-Datei nach Ende der Epoche."""
        duration = time.time() - self.start_time
        self.epoch_times.append(duration)
        print(f"⏱️ Epoche {epoch + 1} dauerte {duration:.2f} Sekunden")

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, duration])

def predict_and_plot_with_gradcam_vis(model, generator, class_names, last_conv_layer_name, num_images=15):
    """
    Führt Grad-CAM Visualisierung für ein Modell auf einem Datenbatch durch.

    Args:
        model (tf.keras.Model): Das trainierte Modell.
        generator (DirectoryIterator): Generator für Eingabebilder.
        class_names (List[str]): Liste der Klassennamen.
        last_conv_layer_name (str): Name der letzten Conv-Schicht für Grad-CAM.
        num_images (int): Anzahl der Bilder, die visualisiert werden sollen.
    """
    # Hole ein Batch
    images, labels = next(generator)
    predictions = model.predict(images)

    # Eval auf Teilmenge
    loss, acc, topKCategoricalAccuracy = model.evaluate(images[:15], labels[:15], verbose=0)
    print(f"Subset Accuracy: {acc * 100:.2f}%, Loss: {loss:.4f}")

    # GradCAM vorbereiten
    replace2linear = ReplaceToLinear()
    gradcam = Gradcam(model, model_modifier=replace2linear, clone=True)

    plt.figure(figsize=(15, 2 * ceil(num_images)))

    for i in range(num_images):
        image = images[i]
        label_index = np.argmax(predictions[i])
        score = CategoricalScore([label_index])  # Zielklasse

        cam = gradcam(score, image[np.newaxis, ...], penultimate_layer=last_conv_layer_name)
        heatmap = cam[0]  # erste (und einzige) Heatmap

        # Visualisierung: Original + Overlay
        ax = plt.subplot(num_images, 2, 2*i + 1)
        plt.imshow(image)
        true_label = class_names[np.argmax(labels[i])]
        predicted_label = class_names[label_index]
        plt.title(f"T: {true_label}\nP: {predicted_label}", fontsize=8)
        plt.axis("off")

        ax = plt.subplot(num_images, 2, 2*i + 2)
        plt.imshow(image)
        plt.imshow(heatmap, cmap='jet', alpha=0.4)
        plt.title("Grad-CAM", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()