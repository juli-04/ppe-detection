"""
SkyGuard - Backend industrial de detección de EPP (YOLOv8 + FastAPI + SQLite)

Ejecutar (desde la raíz del proyecto):
    uvicorn api:app --reload

Endpoints:
    GET  /estado          -> último registro guardado en seguridad.db
    POST /iniciar         -> lanza el bucle de video en un hilo en segundo plano
    POST /detener         -> detiene el bucle de video
    GET  /motor           -> estado del motor de visión (ejecutando, finalizado, error)
    GET  /video_feed      -> streaming MJPEG del video procesado (cajas, etiquetas e IDs)
    POST /fuente/camara   -> cambia a cámara en vivo (?indice=N, p. ej. Iriun Webcam)
    POST /fuente/video    -> sube un .mp4 y lo reproduce
    GET  /camaras         -> lista los índices de cámara disponibles (motor detenido)

Dependencias:
    pip install fastapi uvicorn ultralytics opencv-python-headless python-multipart
"""

import os
import re
import shutil
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, closing
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIGURACIÓN (se puede sobrescribir con variables de entorno)
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("SKYGUARD_DB", "seguridad.db")
MODELO_PATH = os.getenv("SKYGUARD_MODELO", "modelos/best.pt")
FUENTE_POR_DEFECTO = os.getenv("SKYGUARD_FUENTE", "videos_og/prueba.mp4")

CONFIANZA = 0.45
MARGEN_X = 0.20  # 20% del ancho a cada lado
MARGEN_Y = 0.15  # 15% del alto arriba y abajo

INTERVALO_FRAMES = int(os.getenv("SKYGUARD_INTERVALO_FRAMES", "150"))  # registro periódico
FRAMES_ESTABLES = int(os.getenv("SKYGUARD_FRAMES_ESTABLES", "5"))  # anti-parpadeo del estado

CLASES_CONTADAS = ("person", "helmet", "vest")

# --- Streaming ---
VIDEOS_DIR = Path("videos_subidos")
VIDEOS_DIR.mkdir(exist_ok=True)
JPEG_QUALITY = 80   # 60-90: menor = menos ancho de banda
MAX_ANCHO = 960     # el frame se reduce solo para el navegador (la inferencia usa el original)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# clase del modelo -> (etiqueta visible, color BGR)
CLASES_DIBUJO = {
    "person": ("Persona", (255, 128, 0)),   # azul
    "helmet": ("Casco",   (0, 200, 0)),     # verde
    "vest":   ("Chaleco", (0, 165, 255)),   # naranja
}

# ---------------------------------------------------------------------------
# BASE DE DATOS (SQLite)
# ---------------------------------------------------------------------------
def conectar() -> sqlite3.Connection:
    """Abre una conexión nueva. SQLite no debe compartir conexiones entre hilos."""
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")  # lecturas de la API sin bloquear al hilo de video
    return con


def init_db() -> None:
    with closing(conectar()) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS detecciones (
                id        INTEGER PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                frame     INTEGER  NOT NULL,
                estado    TEXT     NOT NULL,
                personas  INTEGER  NOT NULL,
                cascos    INTEGER  NOT NULL,
                chalecos  INTEGER  NOT NULL
            )
            """
        )
        con.commit()


def insertar_deteccion(con: sqlite3.Connection, frame: int, estado: str,
                       personas: int, cascos: int, chalecos: int) -> None:
    con.execute(
        "INSERT INTO detecciones (timestamp, frame, estado, personas, cascos, chalecos) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(sep=" ", timespec="seconds"),
         frame, estado, personas, cascos, chalecos),
    )
    con.commit()


# ---------------------------------------------------------------------------
# LÓGICA DE VISIÓN
# ---------------------------------------------------------------------------
def build_zona_riesgo(ancho: int, alto: int) -> np.ndarray:
    """Rectángulo central adaptativo a la resolución del video."""
    x1, x2 = int(ancho * MARGEN_X), int(ancho * (1 - MARGEN_X))
    y1, y2 = int(alto * MARGEN_Y), int(alto * (1 - MARGEN_Y))
    zona = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    return zona.reshape((-1, 1, 2))


def contar_en_zona(cajas, zona: np.ndarray, nombres: dict) -> dict:
    """Cuenta person/helmet/vest SOLO si el centroide de la caja cae dentro de la zona."""
    conteo = {clase: 0 for clase in CLASES_CONTADAS}

    if cajas.id is not None:
        xyxy = cajas.xyxy.cpu().numpy()
        clases = cajas.cls.int().cpu().tolist()

        for (x1, y1, x2, y2), cls in zip(xyxy, clases):
            cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
            # >= 0: dentro del polígono o sobre el borde
            if cv2.pointPolygonTest(zona, (cx, cy), False) >= 0:
                nombre = nombres[cls]
                if nombre in conteo:
                    conteo[nombre] += 1

    return conteo


def evaluar_estado(personas: int, cascos: int, chalecos: int) -> str:
    falta_casco = cascos < personas
    falta_chaleco = chalecos < personas
    if falta_casco and falta_chaleco:
        return "INCUMPLE"
    if falta_casco:
        return "FALTA CASCO"
    if falta_chaleco:
        return "FALTA CHALECO"
    return "CUMPLE"


# ---------------------------------------------------------------------------
# STREAMING: dibujo y publicación de frames
# ---------------------------------------------------------------------------
def dibujar_detecciones(frame, resultado) -> None:
    """Dibuja cajas, etiquetas e IDs de tracking directamente sobre `frame`."""
    if resultado.boxes is None:
        return
    for caja in resultado.boxes:
        clase = resultado.names[int(caja.cls[0])]
        if clase not in CLASES_DIBUJO:
            continue
        etiqueta, color = CLASES_DIBUJO[clase]
        x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())
        conf = float(caja.conf[0])
        track_id = int(caja.id[0]) if caja.id is not None else None

        texto = etiqueta + (f" #{track_id}" if track_id is not None else "") + f" {conf:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if clase == "person" else 2)

        (tw, th), _ = cv2.getTextSize(texto, FONT, 0.55, 1)
        y_txt = max(y1, th + 8)  # evita que la etiqueta salga del frame
        cv2.rectangle(frame, (x1, y_txt - th - 8), (x1 + tw + 6, y_txt), color, -1)
        cv2.putText(frame, texto, (x1 + 3, y_txt - 4), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def dibujar_banner_estado(frame, estado: str, personas: int, cascos: int, chalecos: int) -> None:
    """Franja superior con el estado global y los conteos dentro de la zona."""
    color = (0, 160, 0) if estado == "CUMPLE" else (0, 0, 220)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), color, -1)
    texto = f"{estado}   Personas: {personas}  Cascos: {cascos}  Chalecos: {chalecos}"
    cv2.putText(frame, texto, (12, 24), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


class Publicador:
    """Guarda el último JPEG y despierta a los clientes MJPEG conectados."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._seq = 0
        self.mensaje("Sin señal", "Inicia la cámara o carga un video")

    def publicar(self, frame) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def mensaje(self, titulo: str, detalle: str = "") -> None:
        """Imagen de relleno cuando no hay video (detenido, error, fin de video)."""
        img = np.full((480, 854, 3), 34, np.uint8)
        for texto, y, escala in ((titulo, 230, 1.0), (detalle[:70], 275, 0.6)):
            if texto:
                (tw, _), _ = cv2.getTextSize(texto, FONT, escala, 2)
                cv2.putText(img, texto, (max((854 - tw) // 2, 5), y), FONT, escala,
                            (200, 200, 200), 2, cv2.LINE_AA)
        self.publicar(img)

    def generar_mjpeg(self):
        """Reenvía el último frame cada 1 s si no hay nuevos (así se detecta al cliente desconectado)."""
        ultimo = -1
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._seq != ultimo, timeout=1.0)
                jpeg, ultimo = self._jpeg, self._seq
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")


publicador = Publicador()


def abrir_captura(fuente) -> cv2.VideoCapture:
    """Cámaras (int) en Windows con DirectShow: abre más rápido y funciona con Iriun."""
    if isinstance(fuente, int) and os.name == "nt":
        return cv2.VideoCapture(fuente, cv2.CAP_DSHOW)
    return cv2.VideoCapture(fuente)


# ---------------------------------------------------------------------------
# MOTOR (hilo en segundo plano)
# ---------------------------------------------------------------------------
class Motor:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.hilo: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.estado = "detenido"  # detenido | iniciando | cargando | ejecutando | finalizado | error
        self.fuente: Optional[str] = None
        self.frames_procesados = 0
        self.error: Optional[str] = None

    def activo(self) -> bool:
        return self.hilo is not None and self.hilo.is_alive()

    def actualizar(self, **campos) -> None:
        with self.lock:
            for k, v in campos.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "estado": self.estado,
                "activo": self.activo(),
                "fuente": self.fuente,
                "frames_procesados": self.frames_procesados,
                "error": self.error,
            }


motor = Motor()


def bucle_video(fuente, stop_event: threading.Event) -> None:
    """Bucle 'ciego': sin imshow, sin waitKey, sin calibración manual. Publica cada frame por MJPEG."""
    cap = None
    con = None
    estado_final = "finalizado"
    try:
        motor.actualizar(estado="cargando")
        # Modelo nuevo por ejecución: evita arrastrar el estado del tracker de un video anterior
        modelo = YOLO(MODELO_PATH)
        nombres = modelo.names

        cap = abrir_captura(fuente)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")

        # Solo los archivos se pacean a su FPS original; cámaras y streams van a la velocidad que entregan
        es_archivo = isinstance(fuente, str) and Path(fuente).is_file()
        intervalo = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30) if es_archivo else 0

        con = conectar()  # conexión propia de este hilo
        motor.actualizar(estado="ejecutando")

        zona = None
        frame_idx = 0
        ultimo_estado: Optional[str] = None
        ultimo_frame_guardado = 0
        candidato: Optional[str] = None
        racha = 0

        while not stop_event.is_set():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break  # fin del video / cámara desconectada
            frame_idx += 1

            # Geofencing estático adaptativo: se calcula una sola vez con el primer frame
            if zona is None:
                alto, ancho = frame.shape[:2]
                zona = build_zona_riesgo(ancho, alto)

            resultados = modelo.track(frame, persist=True, tracker='botsort.yaml',
                                      verbose=False, conf=CONFIANZA)
            cajas = resultados[0].boxes

            conteo = contar_en_zona(cajas, zona, nombres)
            personas, cascos, chalecos = conteo["person"], conteo["helmet"], conteo["vest"]
            estado = evaluar_estado(personas, cascos, chalecos)

            # ¿Cambió el estado de forma estable? (evita spam por parpadeo de detecciones)
            if ultimo_estado is None:
                cambio_confirmado = True  # primer frame: siempre se registra
            elif estado != ultimo_estado:
                racha = racha + 1 if estado == candidato else 1
                candidato = estado
                cambio_confirmado = racha >= FRAMES_ESTABLES
            else:
                racha, candidato = 0, None
                cambio_confirmado = False

            registro_periodico = (frame_idx - ultimo_frame_guardado) >= INTERVALO_FRAMES

            if cambio_confirmado or registro_periodico:
                insertar_deteccion(con, frame_idx, estado, personas, cascos, chalecos)
                ultimo_estado = estado
                ultimo_frame_guardado = frame_idx
                racha, candidato = 0, None

            motor.actualizar(frames_procesados=frame_idx)

            # --- Streaming: dibujar sobre el frame y publicarlo (no toca la lógica de BD) ---
            dibujar_detecciones(frame, resultados[0])
            cv2.polylines(frame, [zona], True, (0, 255, 255), 2)  # zona de geofencing
            dibujar_banner_estado(frame, estado, personas, cascos, chalecos)
            if frame.shape[1] > MAX_ANCHO:
                nuevo_alto = int(frame.shape[0] * MAX_ANCHO / frame.shape[1])
                frame = cv2.resize(frame, (MAX_ANCHO, nuevo_alto))
            publicador.publicar(frame)

            pausa = intervalo - (time.time() - t0)
            if pausa > 0:
                time.sleep(pausa)

        if stop_event.is_set():
            estado_final = "detenido"

    except Exception as exc:  # el hilo nunca debe morir en silencio
        estado_final = "error"
        motor.actualizar(error=f"{type(exc).__name__}: {exc}")
    finally:
        if cap is not None:
            cap.release()
        if con is not None:
            con.close()
        if estado_final == "error":
            publicador.mensaje("Error de fuente", motor.error or "")
        elif estado_final == "finalizado":
            publicador.mensaje("Fin del video", "Carga otro video o inicia la cámara")
        else:
            publicador.mensaje("Transmisión detenida")
        motor.actualizar(estado=estado_final)


def detener_y_esperar(timeout: float = 20) -> None:
    """Pide parar y espera a que el hilo termine (necesario antes de cambiar de fuente)."""
    if not motor.activo():
        return
    motor.stop_event.set()
    motor.hilo.join(timeout=timeout)
    if motor.activo():
        raise HTTPException(status_code=409, detail="El motor tardó en detenerse; reintenta en unos segundos")


def arrancar_motor(fuente_txt: str, reemplazar: bool = False) -> None:
    """Valida la fuente y lanza el hilo. Con reemplazar=True detiene antes la fuente actual."""
    fuente = int(fuente_txt) if fuente_txt.isdigit() else fuente_txt

    es_stream = isinstance(fuente, str) and re.match(r"^(rtsp|rtmp|https?)://", fuente)
    if isinstance(fuente, str) and not es_stream and not Path(fuente).is_file():
        raise HTTPException(status_code=400, detail=f"No existe el archivo de video: {fuente}")

    if reemplazar:
        detener_y_esperar()

    with motor.lock:
        if motor.activo():
            raise HTTPException(status_code=409, detail="El motor de visión ya está en ejecución")
        motor.stop_event = threading.Event()
        motor.estado = "iniciando"
        motor.fuente = fuente_txt
        motor.frames_procesados = 0
        motor.error = None
        motor.hilo = threading.Thread(
            target=bucle_video, args=(fuente, motor.stop_event),
            daemon=True, name="skyguard-video",
        )
        motor.hilo.start()


# ---------------------------------------------------------------------------
# API (FastAPI)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()  # crea seguridad.db y la tabla si no existen
    yield
    motor.stop_event.set()
    if motor.hilo is not None:
        motor.hilo.join(timeout=5)


app = FastAPI(title="SkyGuard API", version="3.1", lifespan=lifespan)


class IniciarRequest(BaseModel):
    fuente: Optional[str] = None  # ruta de video, URL rtsp/http o índice de cámara ("0")


@app.get("/estado")
def obtener_estado():
    """Devuelve el último registro de la tabla detecciones."""
    with closing(conectar()) as con:
        fila = con.execute(
            "SELECT id, timestamp, frame, estado, personas, cascos, chalecos "
            "FROM detecciones ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if fila is None:
        raise HTTPException(status_code=404, detail="Todavía no hay registros en seguridad.db")
    return dict(fila)


@app.post("/iniciar", status_code=202)
def iniciar(body: Optional[IniciarRequest] = None):
    """Dispara el bucle de procesamiento en un hilo daemon sin bloquear la API."""
    fuente_txt = body.fuente if body and body.fuente else FUENTE_POR_DEFECTO
    arrancar_motor(fuente_txt)
    return {"mensaje": "Procesamiento iniciado en segundo plano", "fuente": fuente_txt}


@app.post("/detener")
def detener():
    """Pide al hilo de video que termine (se detiene al finalizar el frame actual)."""
    if not motor.activo():
        raise HTTPException(status_code=409, detail="El motor de visión no está en ejecución")
    motor.stop_event.set()
    return {"mensaje": "Deteniendo el motor de visión"}


@app.get("/motor")
def estado_motor():
    """Estado del hilo de video (útil para ver errores como modelo o video no encontrado)."""
    return motor.snapshot()


# --- Streaming y control de fuente para el frontend ---
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        publicador.generar_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.post("/fuente/camara", status_code=202)
def usar_camara(indice: int = Query(0, ge=0)):
    """Cambia a cámara en vivo. Iriun Webcam suele ser el índice 1 o 2 (0 = cámara del equipo)."""
    arrancar_motor(str(indice), reemplazar=True)
    return {"mensaje": f"Cámara {indice} iniciada", "fuente": str(indice)}


@app.post("/fuente/video", status_code=202)
def usar_video(archivo: UploadFile = File(...)):
    nombre = Path(archivo.filename or "").name
    if not nombre.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="El archivo debe ser .mp4")
    detener_y_esperar()  # libera el archivo si estaba en uso (Windows lo bloquea)
    destino = VIDEOS_DIR / nombre
    with open(destino, "wb") as f:
        shutil.copyfileobj(archivo.file, f)
    arrancar_motor(str(destino))
    return {"mensaje": f"Reproduciendo {nombre}", "fuente": str(destino)}


@app.get("/camaras")
def listar_camaras(max_indice: int = Query(5, ge=1, le=10)):
    """Prueba los primeros índices de cámara. OpenCV no da nombres: distingue por resolución."""
    if motor.activo():
        raise HTTPException(status_code=409, detail="Detén el motor antes de buscar cámaras")
    encontradas = []
    for i in range(max_indice):
        cap = abrir_captura(i)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    encontradas.append({"indice": i, "resolucion": f"{frame.shape[1]}x{frame.shape[0]}"})
        finally:
            cap.release()
    return {"camaras": encontradas}


# ---------------------------------------------------------------------------
# FRONTEND ESTÁTICO: dejar SIEMPRE al final del archivo (después de todas las rutas)
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")