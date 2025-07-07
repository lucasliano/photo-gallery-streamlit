import streamlit as st
import os
from PIL import Image
import sqlite3
from datetime import datetime

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
def save_metadata(filename, author, title):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    hora_actual = datetime.now().strftime("%d/%m/%Y %H:%M")
    c.execute('''
        INSERT INTO photos (filename, author, title, date_uploaded)
        VALUES (?, ?, ?, ?)
    ''', (filename, author, title, hora_actual))
    conn.commit()
    conn.close()

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
# st.title("")
st.markdown("<h1 style='text-align: center;'>Cumple 28 🎂</h1>", unsafe_allow_html=True)


init_db()

# --- Subida desde cámara o galería ---
with st.expander("📤 Subir una imagen (cámara o galería)"):
    st.markdown("### Elegí una imagen")
    image_file = st.file_uploader("📁 Desde tu galería", type=["jpg", "jpeg", "png"])
    # image_camera = st.camera_input("📷 O sacá una foto")

    st.markdown("### Completá los datos")
    author = st.text_input("Autor")
    title = st.text_input("Título")

    image = image_file #or image_camera  # Usar cualquiera

    if image and author and title:
        if st.button("Subir"):
            # Generar nombre único
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            filepath = os.path.join(IMAGES_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(image.getbuffer())
            save_metadata(filename, author, title)
            st.success("Imagen subida correctamente")

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
