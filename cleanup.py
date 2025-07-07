import os
import shutil

# Configuración
DB_FILE = "gallery.db"
IMAGES_DIR = "images"

# Borrar base de datos
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)
    print(f"🗑️ Base de datos '{DB_FILE}' eliminada.")
else:
    print(f"⚠️ Base de datos '{DB_FILE}' no existe.")

# Borrar archivos de imagen
if os.path.exists(IMAGES_DIR):
    files = os.listdir(IMAGES_DIR)
    for f in files:
        path = os.path.join(IMAGES_DIR, f)
        try:
            os.remove(path)
            print(f"🗑️  Imagen eliminada: {f}")
        except Exception as e:
            print(f"⚠️  Error eliminando {f}: {e}")
else:
    print(f"⚠️  Carpeta de imágenes '{IMAGES_DIR}' no existe.")

print("✅ Limpieza completada.")
