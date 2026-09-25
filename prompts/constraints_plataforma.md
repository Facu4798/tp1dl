## Constraints — plataforma, reproducibilidad y presentación de resultados (comunes a los 3 notebooks)

- Todo debe correr en **Google Colab con GPU T4 (16 GB) y RAM estándar** (sin RAM extendida).
- **Reproducibilidad**: fijar la semilla aleatoria (random, numpy, torch, y `transformers.set_seed` donde aplique), imprimir las versiones de las librerías usadas y reportar los hiperparámetros utilizados (dict/tabla impresa).
- **Entrenamiento**: entrenar hasta que la métrica de validación se estabilice (early stopping) y mostrar la curva de entrenamiento/validación.
- **No hardcodear resultados en markdown.** Las celdas markdown contienen solo teoría, metodología y explicación de lo que se va a hacer. Todo número, comparación o conclusión se imprime desde celdas de código con f-strings calculadas sobre las métricas reales (quién gana, deltas, ratios, porcentajes).
- Notebooks en **español**, con el estilo de las prácticas de la cátedra (secciones numeradas, comentarios breves, tablas con pandas, gráficos con matplotlib).
- Checkpoints / artefactos en Google Drive bajo `MyDrive/Colab_Notebooks/TP1_Adell_Giannice_Mazzola/ej{1,2,3}/`, con fallback a una carpeta local cuando no hay Drive (patrón `try: drive.mount(...)` de la cátedra).
- Cada notebook tiene un flag `SMOKE_TEST` (default `False`) en la celda de configuración. Con `True` se achican datos y épocas (subset diminuto, 1 época, sin fp16 si no hay GPU) para que el notebook corra de punta a punta en CPU local; con `False` usa la configuración completa de la consigna. Todo el código debe funcionar con y sin CUDA (device-agnostic). Las celdas `%pip install` / `drive.mount` solo deben ejecutarse en Colab (detectar Colab, p. ej. `'google.colab' in sys.modules` o try/except).
