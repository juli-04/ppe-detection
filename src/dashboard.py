import streamlit as st
import cv2
import pandas as pd
import os
import winsound
from winotify import Notification
from ultralytics import YOLO

# --- Configuración de la página ---
st.set_page_config(page_title="SkyGuard - EPP Detection", layout="wide")
st.title("🦺 SkyGuard - Detección de EPP en Obra")

# --- Cargar el modelo (una sola vez, se cachea) ---
@st.cache_resource
def cargar_modelo():
    return YOLO('modelos/best.pt')

modelo = cargar_modelo()
CLASES = modelo.names

# --- Preparar CSV ---
os.makedirs('reportes', exist_ok=True)
ruta_csv = 'reportes/detecciones.csv'
if not os.path.exists(ruta_csv):
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        f.write('sep=,\n')
        f.write('fecha_hora,frame,estado,personas,cascos,chalecos\n')

# --- Sidebar: elegir fuente ---
st.sidebar.header("Configuración")
modo = st.sidebar.radio("Fuente de video:", ["Video pregrabado", "Cámara en vivo"])

if modo == "Video pregrabado":
    archivo = st.sidebar.file_uploader("Subí un video (.mp4)", type=["mp4"])
    numero_camara = None
else:
    numero_camara = st.sidebar.number_input("Número de cámara", min_value=0, max_value=5, value=0, step=1)
    archivo = None

iniciar = st.sidebar.button("▶ Iniciar detección")
detener = st.sidebar.button("⏹ Detener")

# --- Layout principal ---
col1, col2 = st.columns([2, 1])
with col1:
    video_placeholder = st.empty()
    estado_placeholder = st.empty()
with col2:
    st.subheader("📊 Contadores")
    contadores_placeholder = st.empty()

st.subheader("📋 Historial de incumplimientos")
tabla_placeholder = st.empty()


def notificar_incumplimiento(mensaje):
    winsound.Beep(1000, 300)
    toast = Notification(app_id="SkyGuard EPP", title="⚠️ Incumplimiento de seguridad",
                          msg=mensaje, duration="short")
    toast.show()


def registrar_en_csv(frame_num, estado, conteo):
    with open(ruta_csv, 'a', newline='', encoding='utf-8') as f:
        f.write(f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')},{frame_num},{estado},"
                f"{conteo['person']},{conteo['helmet']},{conteo['vest']}\n")


# --- Sesión para controlar el estado de "detener" ---
if "corriendo" not in st.session_state:
    st.session_state.corriendo = False

if iniciar:
    st.session_state.corriendo = True
if detener:
    st.session_state.corriendo = False

# --- Bucle principal de detección ---
if st.session_state.corriendo:
    if modo == "Video pregrabado" and archivo is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(archivo.read())
        video = cv2.VideoCapture("temp_video.mp4")
    elif modo == "Cámara en vivo":
        video = cv2.VideoCapture(int(numero_camara))
    else:
        st.warning("Subí un video o elegí una cámara antes de iniciar.")
        video = None

    if video is not None and video.isOpened():
        incumplimiento_previo = False
        numero_frame = 0
        total_incumplimientos = 0

        while video.isOpened() and st.session_state.corriendo:
            ret, frame = video.read()
            if not ret:
                break

            numero_frame += 1
            resultados = modelo(frame, verbose=False, conf=0.7)
            frame_anotado = resultados[0].plot()

            conteo = {"head": 0, "helmet": 0, "person": 0, "vest": 0}
            for caja in resultados[0].boxes:
                nombre_clase = CLASES[int(caja.cls[0])]
                conteo[nombre_clase] += 1

            hay_incumplimiento = False
            if conteo["person"] > 0:
                if conteo["helmet"] > 0 and conteo["vest"] > 0:
                    estado = "✅ CUMPLE (casco + chaleco)"
                elif conteo["helmet"] > 0:
                    estado = "⚠️ FALTA CHALECO"
                    hay_incumplimiento = True
                elif conteo["vest"] > 0:
                    estado = "⚠️ FALTA CASCO"
                    hay_incumplimiento = True
                else:
                    estado = "🚨 INCUMPLE (sin casco ni chaleco)"
                    hay_incumplimiento = True
            else:
                estado = "Sin personas detectadas"

            if hay_incumplimiento and not incumplimiento_previo:
                notificar_incumplimiento(estado)
                registrar_en_csv(numero_frame, estado, conteo)
                total_incumplimientos += 1
            incumplimiento_previo = hay_incumplimiento

            # Mostrar video (convertir BGR a RGB para Streamlit)
            frame_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")
            estado_placeholder.markdown(f"### Estado actual: {estado}")

            contadores_placeholder.markdown(f"""
            - **Frames procesados:** {numero_frame}
            - **Incumplimientos detectados:** {total_incumplimientos}
            - **Personas en frame:** {conteo['person']}
            """)

            # Refrescar tabla cada 20 frames (para no sobrecargar)
            if numero_frame % 20 == 0 and os.path.exists(ruta_csv):
                df = pd.read_csv(ruta_csv, skiprows=1)
                tabla_placeholder.dataframe(df.tail(10), use_container_width=True)

        video.release()
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
else:
    st.info("Elegí una fuente de video y presioná ▶ Iniciar detección.")