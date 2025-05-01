import os
import unicodedata
import re

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