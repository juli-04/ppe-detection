from ultralytics import YOLO

print("Cargando el modelo SkyGuard...")
# 1. Cargamos tu modelo entrenado
modelo = YOLO("modelos/best.pt")

print("Iniciando evaluación de metricas. Esto puede demorar unos minutos...")
# 2. Ejecutamos la validación. 

resultados = modelo.val(
    data="dataset_epp/data.yaml",  # Archivo de configuración del dataset
    save=True,            # Guardamos los resultados de la evaluación en un archivo
    plots=True,           # Le pedimos que genere todos los gráficos (PNG)
    batch=16,             # Tamaño del lote (ajustable según tu RAM)
    conf=0.25             # Umbral de confianza estándar para métricas
)

print("\n"+"="*50)
print("¡EVALUACIÓN COMPLETADA!")
print("="*50)
print(f"mAP@50 (Precisión Media): {resultados.box.map50:.4f}")
print(f"mAP@50-95 (Precisión Estricta): {resultados.box.map:.4f}")
print("\nTodos los gráficos (Matriz de Confusión, Curvas F1, etc.)")
print("se guardaron automáticamente en la carpeta: 'runs/detect/val'")