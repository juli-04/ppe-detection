import sys
import cv2
import numpy as np
import csv
import os
from datetime import datetime
from ultralytics import YOLO

# 1. Cargar el modelo entrenado por nosotros
modelo = YOLO('modelos/best.pt')
print(modelo.names)  # {0: 'head', 1: 'helmet', 2: 'person', 3: 'vest'}

CLASES = modelo.names

# 2. Definir la fuente de video (archivo o cámara en vivo)
# Uso:
#   python src/main.py                          -> usa video por defecto
#   python src/main.py videos_og/otro_video.mp4  -> usa ese archivo
#   python src/main.py 1                         -> usa la cámara número 1 (en vivo)

fuente = sys.argv[1] if len(sys.argv) > 1 else 'videos_og/prueba2.mp4'
es_camara = fuente.isdigit()

if es_camara:
    video = cv2.VideoCapture(int(fuente))
else:
    video = cv2.VideoCapture(fuente)

if not video.isOpened():
    print("Error: no se pudo abrir la fuente de video. Revisá la ruta o el número de cámara.")
    exit()

# Geofencing Interactivo: Calibración manual de la Zona de Riesgo
# Se toma el primer frame como referencia (mínimo 3 vértices).

puntos_zona = []


def click_calibracion(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Vertice agregado: ({x}, {y})")
        puntos_zona.append((x, y))


ret, primer_frame = video.read()
if not ret:
    print("Error: no se pudo leer el primer frame para la calibración.")
    exit()

cv2.namedWindow("Calibracion", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Calibracion", 960, 540) # Fuerza el tamaño a la mitad de 1080p
cv2.setMouseCallback("Calibracion", click_calibracion)

while True:
    frame_calibracion = primer_frame.copy()

    for punto in puntos_zona:
        cv2.circle(frame_calibracion, punto, 15, (0, 255, 0), -1) # Subimos el radio a 15

    if len(puntos_zona) > 1:
        cv2.polylines(frame_calibracion, [np.array(puntos_zona, dtype=np.int32)],
                       isClosed=False, color=(0, 0, 255), thickness=5) # Subimos el grosor a 5
        
    cv2.putText(frame_calibracion, "Clic izquierdo: agregar vertice | ENTER: confirmar (min. 3 puntos)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Calibracion", frame_calibracion)
    tecla = cv2.waitKey(20) & 0xFF

    if tecla == 13:  # Enter
        if len(puntos_zona) >= 3:
            break
        else:
            print("Se necesitan al menos 3 vertices para definir la zona de riesgo.")

cv2.destroyWindow("Calibracion")

ZONA_RIESGO = np.array(puntos_zona, dtype=np.int32)

# Si la fuente es un archivo, se vuelve al frame 0 para no perder el primer
# frame (ya consumido durante la calibración) del procesamiento principal.
# En cámara en vivo esto no aplica: no se puede "rebobinar" el stream.
if not es_camara:
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- Preparar el archivo CSV de reporte ---
os.makedirs('reportes', exist_ok=True)
ruta_csv = 'reportes/detecciones.csv'

# Si el archivo no existe, lo creamos con encabezados
if not os.path.exists(ruta_csv):
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        f.write('sep=,\n')
        writer = csv.writer(f)
        writer.writerow(['fecha_hora', 'frame', 'estado', 'personas', 'cascos', 'chalecos'])

# Variable para saber si YA estábamos en incumplimiento (evita notificar en cada frame)
incumplimiento_previo = False
numero_frame = 0


def notificar_incumplimiento(mensaje):
    """Envía la alerta de incumplimiento.
    Por ahora solo la imprime en consola (simulando el envío).
    Fase siguiente: acá va un requests.post hacia la API de Grafana,
    por ejemplo: requests.post(GRAFANA_URL, json={"mensaje": mensaje, "timestamp": ...})
    """
    print(f"ALERTA ENVIADA AL SERVIDOR: {mensaje}")


def registrar_en_csv(frame_num, estado, conteo):
    with open(ruta_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            frame_num,
            estado,
            conteo['person'],
            conteo['helmet'],
            conteo['vest']
        ])


def centroide(x1, y1, x2, y2):
    """Centroide de la caja delimitadora (x_c, y_c), como floats."""
    return (x1 + x2) / 2, (y1 + y2) / 2


# 3. Procesar frame por frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    numero_frame += 1

    # Tracking nativo (BoT-SORT).
    resultados = modelo.track(frame, persist=True, tracker='botsort.yaml', verbose=False, conf=0.5)
    frame_anotado = resultados[0].plot()

    # Dibujar la zona de riesgo calibrada sobre el frame anotado
    pts_zona = ZONA_RIESGO.reshape((-1, 1, 2))
    cv2.polylines(frame_anotado, [pts_zona], isClosed=True, color=(0, 0, 255), thickness=2)

    cajas = resultados[0].boxes
    ids = cajas.id  # None si todavía no hay tracks confirmados en este frame (o no hay detecciones)

    # --- Conteo SOLO dentro de la zona de riesgo ---
    conteo = {"head": 0, "helmet": 0, "person": 0, "vest": 0}

    if ids is not None:
        ids = ids.int().cpu().tolist()
        xyxy = cajas.xyxy.cpu().tolist()
        clases_id = cajas.cls.cpu().tolist()

        for (x1, y1, x2, y2), clase_id, id_seguimiento in zip(xyxy, clases_id, ids):
            nombre_clase = CLASES[int(clase_id)]
            xc, yc = centroide(x1, y1, x2, y2)

            dentro_zona = cv2.pointPolygonTest(ZONA_RIESGO, (xc, yc), False) >= 0

            # Marcar el centroide: verde si está dentro de la zona, gris si está fuera
            color_punto = (0, 255, 0) if dentro_zona else (150, 150, 150)
            cv2.circle(frame_anotado, (int(xc), int(yc)), 4, color_punto, -1)

            # Restricción de negocio: los objetos fuera de la ZONA_RIESGO se ignoran
            # por completo para el conteo y para la lógica de cumplimiento.
            if dentro_zona:
                conteo[nombre_clase] += 1
            # id_seguimiento queda disponible para la Fase 2 (historial/suavizado por ID)

    # --- Lógica de cumplimiento (misma que antes, ahora solo sobre lo que está en zona) ---
    hay_incumplimiento = False
    if conteo["person"] > 0:
        if conteo["helmet"] > 0 and conteo["vest"] > 0:
            estado = "CUMPLE (casco + chaleco)"
            color = (0, 255, 0)
        elif conteo["helmet"] > 0:
            estado = "FALTA CHALECO"
            color = (0, 165, 255)
            hay_incumplimiento = True
        elif conteo["vest"] > 0:
            estado = "FALTA CASCO"
            color = (0, 165, 255)
            hay_incumplimiento = True
        else:
            estado = "INCUMPLE (sin casco ni chaleco)"
            color = (0, 0, 255)
            hay_incumplimiento = True
    else:
        estado = "Sin personas en zona de riesgo"
        color = (255, 255, 255)

    # Notificar y registrar SOLO cuando empieza un nuevo incumplimiento
    if hay_incumplimiento and not incumplimiento_previo:
        notificar_incumplimiento(estado)
        registrar_en_csv(numero_frame, estado, conteo)
    incumplimiento_previo = hay_incumplimiento

    cv2.putText(frame_anotado, estado, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    frame_mostrar = cv2.resize(frame_anotado, (960, 540))
    cv2.imshow('SkyGuard - Deteccion EPP', frame_mostrar)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print(f"\nReporte guardado en: {ruta_csv}")