import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import winsound
from datetime import datetime
from winotify import Notification
from ultralytics import YOLO

# ==================== CONFIGURACIÓN DE PÁGINA ====================
st.set_page_config(page_title="SkyGuard | Sistema de Seguridad Industrial",
                    layout="wide", initial_sidebar_state="expanded")

# ==================== ESTILO VISUAL PROFESIONAL ====================
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stApp { background-color: #0E1117; }
    h1, h2, h3 { color: #F0F2F6; font-family: 'Segoe UI', sans-serif; }
    .titulo-app {
        font-size: 32px; font-weight: 700; color: #FFFFFF;
        border-bottom: 3px solid #2563EB; padding-bottom: 10px; margin-bottom: 20px;
    }
    .subtitulo { color: #9CA3AF; font-size: 14px; margin-bottom: 30px; }
    .badge-estado {
        padding: 18px; border-radius: 8px; text-align: center;
        font-size: 22px; font-weight: 700; color: white; margin-bottom: 20px;
    }
    .card {
        background-color: #1A1D24; padding: 20px; border-radius: 10px;
        border: 1px solid #2D3139; margin-bottom: 15px;
    }
    .card-titulo { color: #9CA3AF; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
    .card-valor { color: #FFFFFF; font-size: 28px; font-weight: 700; }
    .stButton>button {
        background-color: #2563EB; color: white; border: none;
        border-radius: 6px; padding: 10px 20px; font-weight: 600;
    }
    .stButton>button:hover { background-color: #1D4ED8; }
</style>
""", unsafe_allow_html=True)

# ==================== USUARIOS (demo) ====================
USUARIOS = {
    "inspector": {"password": "1234", "rol": "Inspector", "nombre": "Julieta Leo"},
    "gerente":   {"password": "1234", "rol": "Gerente de Obra", "nombre": "Gerente General"},
}

# ==================== ESTADO DE SESIÓN ====================
if "logueado" not in st.session_state:
    st.session_state.logueado = False
if "corriendo" not in st.session_state:
    st.session_state.corriendo = False

# ==================== PANTALLA DE LOGIN ====================
def pantalla_login():
    st.markdown('<div class="titulo-app">SkyGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitulo">Sistema de Monitoreo de Seguridad Industrial</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("### Iniciar sesión")
        usuario = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        if st.button("Ingresar", use_container_width=True):
            if usuario in USUARIOS and USUARIOS[usuario]["password"] == password:
                st.session_state.logueado = True
                st.session_state.usuario = usuario
                st.session_state.rol = USUARIOS[usuario]["rol"]
                st.session_state.nombre = USUARIOS[usuario]["nombre"]
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

# ==================== FUNCIONES AUXILIARES ====================
def notificar_incumplimiento(mensaje):
    winsound.Beep(1000, 300)
    toast = Notification(app_id="SkyGuard", title="Incumplimiento de seguridad detectado",
                          msg=mensaje, duration="short")
    toast.show()

def registrar_en_csv(ruta_csv, frame_num, estado, conteo, usuario):
    with open(ruta_csv, 'a', newline='', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{frame_num},{estado},"
                f"{conteo['person']},{conteo['helmet']},{conteo['vest']},{usuario}\n")

def obtener_estado(conteo):
    """Devuelve estado y color según la lógica de semáforo."""
    if conteo["person"] == 0:
        return "SIN PERSONAS DETECTADAS", "#4B5563"
    tiene_helmet = conteo["helmet"] > 0
    tiene_vest = conteo["vest"] > 0
    if tiene_helmet and tiene_vest:
        return "CUMPLE - CASCO Y CHALECO", "#16A34A"       # verde
    elif tiene_helmet or tiene_vest:
        return "CUMPLIMIENTO PARCIAL", "#CA8A04"           # amarillo
    else:
        return "INCUMPLIMIENTO TOTAL", "#DC2626"           # rojo

def build_zona_riesgo(ancho, alto):
    """
    Geofencing estático adaptativo: polígono rectangular de 4 vértices centrado,
    con margen del 20% en el ancho y 15% en el alto. No requiere clics.
    """
    margen_x = int(ancho * 0.20)
    margen_y = int(alto * 0.15)
    return np.array([
        [margen_x,         margen_y],           # superior izquierdo
        [ancho - margen_x, margen_y],           # superior derecho
        [ancho - margen_x, alto - margen_y],    # inferior derecho
        [margen_x,         alto - margen_y],    # inferior izquierdo
    ], dtype=np.int32)

@st.cache_resource
def cargar_modelo():
    return YOLO('modelos/best.pt')

# ==================== VISTA: INSPECTOR ====================
def vista_inspector():
    modelo = cargar_modelo()
    CLASES = modelo.names

    os.makedirs('reportes', exist_ok=True)
    ruta_csv = 'reportes/detecciones.csv'
    if not os.path.exists(ruta_csv):
        with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
            f.write('sep=,\n')
            f.write('fecha_hora,frame,estado,personas,cascos,chalecos,usuario\n')

    st.sidebar.markdown("### Configuración")
    modo = st.sidebar.radio("Fuente de video", ["Video pregrabado", "Cámara en vivo"])

    if modo == "Video pregrabado":
        archivo = st.sidebar.file_uploader("Subir video (.mp4)", type=["mp4"])
        numero_camara = None
    else:
        numero_camara = st.sidebar.number_input("Número de cámara", min_value=0, max_value=5, value=0)
        archivo = None

    iniciar = st.sidebar.button("Iniciar detección", use_container_width=True)
    detener = st.sidebar.button("Detener", use_container_width=True)

    if iniciar:
        st.session_state.corriendo = True
    if detener:
        st.session_state.corriendo = False

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="titulo-app" style="font-size:22px;">Monitoreo en tiempo real</div>', unsafe_allow_html=True)
        video_placeholder = st.empty()
        estado_placeholder = st.empty()
    with col2:
        st.markdown('<div class="titulo-app" style="font-size:22px;">Indicadores</div>', unsafe_allow_html=True)
        contadores_placeholder = st.empty()

    st.markdown('<div class="titulo-app" style="font-size:22px;">Historial de eventos</div>', unsafe_allow_html=True)
    tabla_placeholder = st.empty()

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

            # Geofencing: se calcula una sola vez con las dimensiones del primer frame
            ZONA_RIESGO = None

            # El modelo está cacheado (@st.cache_resource): se reinicia el predictor
            # para que el tracker BoT-SORT no arrastre IDs de una ejecución anterior.
            if getattr(modelo, "predictor", None) is not None:
                modelo.predictor = None

            while video.isOpened() and st.session_state.corriendo:
                ret, frame = video.read()
                if not ret:
                    break

                numero_frame += 1

                # --- Geofencing: cálculo único de ZONA_RIESGO desde frame.shape ---
                if ZONA_RIESGO is None:
                    alto_frame, ancho_frame = frame.shape[:2]
                    ZONA_RIESGO = build_zona_riesgo(ancho_frame, alto_frame)

                # --- Tracking con BoT-SORT ---
                resultados = modelo.track(frame, persist=True, tracker='botsort.yaml',
                                          verbose=False, conf=0.45)
                frame_anotado = resultados[0].plot()

                # Zona de riesgo en rojo (BGR) sobre el frame anotado
                cv2.polylines(frame_anotado, [ZONA_RIESGO], isClosed=True,
                              color=(0, 0, 255), thickness=3)

                # --- Extracción de cajas, clases e IDs ---
                conteo = {"head": 0, "helmet": 0, "person": 0, "vest": 0}
                cajas = resultados[0].boxes

                if cajas is not None and len(cajas) > 0:
                    xyxy = cajas.xyxy.cpu().numpy()
                    clases = cajas.cls.int().cpu().tolist()
                    ids = cajas.id  # None si el tracker aún no asignó IDs
                    if ids is not None:
                        ids = ids.int().cpu().tolist()
                    else:
                        ids = [None] * len(clases)

                    for (x1, y1, x2, y2), cls_id, track_id in zip(xyxy, clases, ids):
                        # Centroide de la caja
                        xc = int((x1 + x2) / 2)
                        yc = int((y1 + y2) / 2)

                        # Filtro espacial: >= 0 significa dentro o sobre el borde
                        dentro = cv2.pointPolygonTest(ZONA_RIESGO, (float(xc), float(yc)), False) >= 0

                        if dentro:
                            # SOLO se cuentan los objetos dentro de la zona
                            conteo[CLASES[cls_id]] += 1
                            color_centroide = (0, 255, 0)      # verde
                        else:
                            color_centroide = (128, 128, 128)  # gris

                        cv2.circle(frame_anotado, (xc, yc), 5, color_centroide, -1)

                        if track_id is not None:
                            cv2.putText(frame_anotado, f"ID {track_id}", (xc + 8, yc - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_centroide, 1, cv2.LINE_AA)

                estado, color = obtener_estado(conteo)
                hay_incumplimiento = color in ("#DC2626", "#CA8A04")

                if hay_incumplimiento and not incumplimiento_previo:
                    notificar_incumplimiento(estado)
                    registrar_en_csv(ruta_csv, numero_frame, estado, conteo, st.session_state.usuario)
                    total_incumplimientos += 1
                incumplimiento_previo = hay_incumplimiento

                frame_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB")
                estado_placeholder.markdown(
                    f'<div class="badge-estado" style="background-color:{color};">{estado}</div>',
                    unsafe_allow_html=True)

                contadores_placeholder.markdown(f"""
                <div class="card"><div class="card-titulo">Frames procesados</div><div class="card-valor">{numero_frame}</div></div>
                <div class="card"><div class="card-titulo">Incumplimientos</div><div class="card-valor">{total_incumplimientos}</div></div>
                <div class="card"><div class="card-titulo">Personas en cuadro</div><div class="card-valor">{conteo['person']}</div></div>
                """, unsafe_allow_html=True)

                if numero_frame % 20 == 0 and os.path.exists(ruta_csv):
                    df = pd.read_csv(ruta_csv, skiprows=1)
                    tabla_placeholder.dataframe(df.tail(10), use_container_width=True)

            video.release()
            if os.path.exists("temp_video.mp4"):
                os.remove("temp_video.mp4")
    else:
        st.info("Seleccioná una fuente de video y presioná Iniciar detección.")

# ==================== VISTA: GERENTE DE OBRA (solo lectura) ====================
def vista_gerente():
    st.markdown('<div class="titulo-app" style="font-size:22px;">Panel de Reportes</div>', unsafe_allow_html=True)

    ruta_csv = 'reportes/detecciones.csv'
    if not os.path.exists(ruta_csv):
        st.warning("Todavía no hay datos registrados.")
        return

    df = pd.read_csv(ruta_csv, skiprows=1)

    total_eventos = len(df)
    incumplimientos_totales = len(df[df['estado'] == 'INCUMPLIMIENTO TOTAL'])
    incumplimientos_parciales = len(df[df['estado'] == 'CUMPLIMIENTO PARCIAL'])

    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="card"><div class="card-titulo">Eventos registrados</div><div class="card-valor">{total_eventos}</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card"><div class="card-titulo">Incumplimientos totales</div><div class="card-valor" style="color:#DC2626;">{incumplimientos_totales}</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card"><div class="card-titulo">Cumplimientos parciales</div><div class="card-valor" style="color:#CA8A04;">{incumplimientos_parciales}</div></div>', unsafe_allow_html=True)

    st.markdown("### Historial completo")
    st.dataframe(df.sort_values(by="fecha_hora", ascending=False), use_container_width=True)

    if "usuario" in df.columns:
        st.markdown("### Eventos por usuario")
        st.bar_chart(df["usuario"].value_counts())

# ==================== APP PRINCIPAL ====================
if not st.session_state.logueado:
    pantalla_login()
else:
    st.sidebar.markdown(f"*Usuario:* {st.session_state.nombre}")
    st.sidebar.markdown(f"*Rol:* {st.session_state.rol}")
    if st.sidebar.button("Cerrar sesión"):
        st.session_state.logueado = False
        st.session_state.corriendo = False
        st.rerun()
    st.sidebar.markdown("---")

    st.markdown('<div class="titulo-app">SkyGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitulo">Sistema de Monitoreo de Seguridad Industrial</div>', unsafe_allow_html=True)

    if st.session_state.rol == "Inspector":
        vista_inspector()
    else:
        vista_gerente()