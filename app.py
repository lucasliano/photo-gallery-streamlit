from utils.image_processing import process_image
from datetime import datetime
import streamlit as st
from PIL import Image
import sqlite3
import os

# --- Configuración ---
IMAGES_DIR = "images"
DB_FILE = "gallery.db"
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Inicializar base de datos ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            author TEXT,
            title TEXT,
            date_uploaded TEXT
        )
    ''')
    conn.commit()
    conn.close()

# --- Guardar metadata en base de datos ---
def save_metadata(image, author, title):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    hora_actual = datetime.now().strftime("%d/%m/%Y %H:%M")
    c.execute('''
        INSERT INTO photos (filename, author, title, date_uploaded)
        VALUES (?, ?, ?, ?)
    ''', ("", author, title, hora_actual))  # Filename se actualiza luego
    photo_id = c.lastrowid
    conn.commit()
    
    filename = f"{photo_id}.jpg"         # Guardamos la foto con el ID de la db
    filepath = os.path.join(IMAGES_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image.getbuffer())

    c.execute('''
        UPDATE photos SET filename = ? WHERE id = ?
    ''', (filename, photo_id))
    conn.commit()
    conn.close()

    return filepath

# --- Obtener metadata ---
def get_photos():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT filename, author, title, date_uploaded FROM photos ORDER BY id DESC')
    photos = c.fetchall()
    conn.close()
    return photos

# --- Contador de fotos ---
def count_photos():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM photos')
    count = c.fetchone()[0]
    conn.close()
    return count

# --- Interfaz Streamlit ---
st.set_page_config(page_title="Galería", layout="wide")
st.markdown("<h1 style='text-align: center;'> Generador de Dataset</h1>", unsafe_allow_html=True)

init_db()

# --- Subida desde cámara o galería ---
with st.expander("📤 Subir una imagen (cámara o galería)"):
    st.markdown("### Elegí una imagen")
    image = st.file_uploader("📁 Desde tu galería", type=["jpg", "jpeg", "png"])

    st.markdown("### Completá los datos")
    author = st.text_input("Autor")
    title = st.text_input("Título")

    if image and author and title:
        if st.button("Subir"):
            try:
                image_filepath = save_metadata(image, author, title)
                result_data = process_image(image_filepath)

                st.success(f"Imagen subida correctamente. Charuco Detected: {result_data['charuco_detected']}, QRs detected: {result_data['qr_codes_json']}")
            except Exception as e:
                st.error(f"Error: {e}")

# --- Mostrar galería ---
st.markdown("## 📸 Galería")
st.markdown(f"¡Ya tenemos **{count_photos()}** fotos! 🎉")
photos = get_photos()
cols = st.columns(3)

for i, (filename, author, title, date_uploaded) in enumerate(photos):
    col = cols[i % 3]
    with col:
        image_path = os.path.join(IMAGES_DIR, filename)
        try:
            img = Image.open(image_path)
            st.image(img, use_container_width=True, caption=f"**{title}** \n\n _{author}, {date_uploaded}_")
        except Exception as e:
            st.error(f"Error mostrando {filename}: {e}")
