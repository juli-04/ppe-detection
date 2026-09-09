import sys
import cv2
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

if fuente.isdigit():
    video = cv2.VideoCapture(int(fuente))
else:
    video = cv2.VideoCapture(fuente)

if not video.isOpened():
    print("Error: no se pudo abrir la fuente de video. Revisá la ruta o el número de cámara.")
    exit()

# 3. Procesar frame por frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    resultados = modelo(frame)
    frame_anotado = resultados[0].plot()

    # --- Lógica de conteo ---
    conteo = {"head": 0, "helmet": 0, "person": 0, "vest": 0}
    for caja in resultados[0].boxes:
        clase_id = int(caja.cls[0])
        nombre_clase = CLASES[clase_id]
        conteo[nombre_clase] += 1

    # --- Lógica de cumplimiento ---
    if conteo["person"] > 0:
        if conteo["helmet"] > 0 and conteo["vest"] > 0:
            estado = "CUMPLE (casco + chaleco)"
            color = (0, 255, 0)  # verde
        elif conteo["helmet"] > 0:
            estado = "FALTA CHALECO"
            color = (0, 165, 255)  # naranja
        elif conteo["vest"] > 0:
            estado = "FALTA CASCO"
            color = (0, 165, 255)
        else:
            estado = "INCUMPLE (sin casco ni chaleco)"
            color = (0, 0, 255)  # rojo
    else:
        estado = "Sin personas detectadas"
        color = (255, 255, 255)

    cv2.putText(frame_anotado, estado, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    frame_mostrar = cv2.resize(frame_anotado, (960, 540))
    cv2.imshow('SkyGuard - Deteccion EPP', frame_mostrar)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()