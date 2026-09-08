import cv2
from ultralytics import YOLO

# 1. Cargar el modelo entrenado por nosotros
modelo = YOLO('modelos/best.pt')

# 2. Abrir el video
video = cv2.VideoCapture('videos_og/prueba.mp4')

if not video.isOpened():
    print("Error: no se pudo abrir el video. Revisá la ruta.")
    exit()

# 3. Procesar frame por frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break # se terminó el video

# Le pasamos el frame al modelo
    resultados = modelo(frame)

# Dibujamos las detecciones sobre el frame
    frame_anotado = resultados[0].plot()

# Redimensionamos para que la ventana no sea gigante
    frame_mostrar = cv2.resize(frame_anotado, (960, 540))

# Mostramos el resultado en una ventana
    cv2.imshow('SkyGuard - Deteccion EPP', frame_mostrar)

# Presionar 'q' para cortar antes de que termine el video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows() 