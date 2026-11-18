# ppe-detection
real time PPE (helmet and safety vest) detection on construction sites using a fine-tuned YOLOv8 model and drone footage
# SkyGuard — Detección de EPP (casco y chaleco) en tiempo real

Sistema de visión por computadora que verifica en tiempo real si las personas en una obra llevan **casco** y **chaleco** de seguridad. Usa un modelo **YOLOv8n** entrenado por el equipo (transfer learning, 50 épocas, imágenes de 640×640), un backend en **FastAPI** con seguimiento de personas (BoT-SORT), control por zona (geofencing) y registro en **SQLite**, y un panel web.

Trabajo final de *Seminario de Actualización II* — Universidad Nacional de La Rioja (UNLaR).

**Integrantes:** de la Fuente Gonzalez, Letizia Delfina · Leo Adriana, Julieta · Vasquez, Maria Jose Ines

## Resultados del modelo (validación final sobre `best.pt`)

| Métrica | Valor |
|---|---|
| Precision | 0.803 |
| Recall | 0.668 |
| mAP50 | 0.796 |
| mAP50-95 | 0.461 |

Por clase (mAP50-95): persona 0.579 · chaleco 0.531 · cabeza 0.426 · **casco 0.307** (la clase más débil).

## Requisitos

- Python 3.10 o superior (con `pip`) y `git`
- Windows, Linux o Mac
- Una cámara web (o el celular con la app Iriun Webcam) **o** un video `.mp4`
- No hace falta GPU: YOLOv8n corre en CPU
- Conexión a internet (el panel carga estilos y tipografías desde la web)

## Instalación

```bash
git clone https://github.com/juli-04/ppe-detection.git
cd ppe-detection

# 1) Entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

# 2) Dependencias (la primera vez tarda unos minutos: descarga PyTorch)
pip install -r requirements.txt
```

## Ejecución

Siempre desde la carpeta del proyecto (`ppe-detection/`), con el entorno activado:

```bash
uvicorn api:app --reload
```

Debe aparecer `Uvicorn running on http://127.0.0.1:8000`. Si el puerto está ocupado: `uvicorn api:app --reload --port 8001`.

## Uso del panel

Abrir <http://127.0.0.1:8000> en el navegador.

- **Cámara en vivo:** elegir el índice en el desplegable (0 = cámara de la computadora; Iriun Webcam suele ser 1 o 2; el botón *Buscar* lista las disponibles) y tocar **Iniciar cámara web**. La zona de control es todo el cuadro.
- **Video grabado:** tocar **Cargar video grabado (MP4)** y elegir el archivo. Se abre la ventana *Calibracion Manual*: clic izquierdo para marcar los puntos de la zona (mínimo 3) y **Enter** para confirmar.
- **Detener** corta el proceso.

## Resultado esperado

- Cajas alrededor de persona, casco y chaleco, con un ID de seguimiento por persona.
- Un banner de estado: verde si **CUMPLE**; rojo si falta casco, chaleco o ambos.
- `GET /estado` devuelve el último registro guardado en `seguridad.db` (hora, personas, cascos y chalecos).

## Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/estado` | Último registro guardado |
| POST | `/iniciar` · `/detener` | Inicia o detiene el motor de visión |
| GET | `/motor` | Estado del motor (ejecutando, finalizado, error) |
| GET | `/video_feed` | Streaming MJPEG del video procesado |
| POST | `/fuente/camara` · `/fuente/video` | Cambia la fuente de video |
| GET | `/camaras` | Lista los índices de cámara disponibles |

Documentación interactiva: <http://127.0.0.1:8000/docs>

## Configuración

`CONFIANZA` (umbral de detección, por defecto 0.45) está en `api.py`. Variables de entorno opcionales: `SKYGUARD_MODELO`, `SKYGUARD_FUENTE`, `SKYGUARD_DB`, `SKYGUARD_INTERVALO_FRAMES`, `SKYGUARD_FRAMES_ESTABLES`.

## Errores frecuentes

| Problema | Solución |
|---|---|
| `uvicorn` no se reconoce / `ModuleNotFoundError` | Activar el entorno virtual e instalar: `pip install -r requirements.txt` |
| El panel dice «No hay conexión con el servidor» | `uvicorn` no está corriendo: volver a ejecutar `uvicorn api:app --reload` |
| Estado «Error» en el panel | Abrir `/motor`: muestra el mensaje exacto (por ejemplo, el archivo de video no existe) |
| La cámara no abre | Probar otro índice (0, 1 o 2) |
| No reconoce el casco | Es la clase más débil del modelo; probar bajar `CONFIANZA` en `api.py` |
| No se abre la ventana de calibración | Instalar `opencv-python` (no la versión `headless`) |

## Estructura

```
ppe-detection/
├── api.py               # backend FastAPI + motor de visión
├── requirements.txt     # dependencias
├── modelos/best.pt      # modelo YOLOv8 entrenado
├── static/index.html    # panel web
├── generar_metrica.py   # evaluación independiente (requiere el dataset local)
├── reportes/            # registros de ejemplo
└── src/                 # versión inicial (prototipo con Streamlit)
```

## Licencia

MIT — ver [LICENSE](LICENSE).
