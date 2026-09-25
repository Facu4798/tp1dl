# Smoke test — Ej. 1 (ESC-50)

- **Resultado:** PASS (nbconvert exit code 0).
- **Notebook ejecutado:** `outputs/smoke/ej1_smoke.ipynb`
- **Tiempo:** ~20 s con la extracción del zip incluida; ~15 s con los caches ya creados (CPU, sin CUDA). El zip de ESC-50 (~646 MB) se había descargado antes a `C:/Users/facum/tp1_cache/ej1/`; el notebook detecta que existe y no lo vuelve a bajar.
- **Comando** (Git Bash, desde la raíz del proyecto):

  ```
  JUPYTER_PATH=C:/Users/facum/tp1_cache/jupyter TP1_SMOKE_TEST=1 TP1_LOCAL_DIR=C:/Users/facum/tp1_cache \
    E:/Programs/miniconda/envs/tp1/python.exe -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=1800 src/Adell_Giannice_Mazzola_Ej1_ESC50.ipynb \
    --output-dir outputs/smoke --output ej1_smoke.ipynb
  ```

## Qué muestra el notebook ejecutado

- Configuración, versiones y `HPARAMS`; split 30/10/10 clips (5 clases × 2 por fold).
- Formas de onda + mel-spectrogramas, STFT lineal vs. mel, resoluciones (Δf = 10.77 Hz, Δt = 23.2 ms, 216 frames) y demo del trade-off con n_fft 256 vs. 4096.
- Tabla de parámetros: CNN 401 650; A 218 610; B 94 066; encoder idéntico (66 944) en A y B; tokens A = 12, B = 112.
- Entrenamiento de 1 época de los tres modelos, curvas, tabla comparativa (accuracy, F1-macro, parámetros, tiempo), tres matrices de confusión 50×50, atención de B (última capa y rollout, aciertos y errores), mapa de atención medio y conclusiones impresas.
- Grep de clases prohibidas (`nn.Transformer*`, `nn.MultiheadAttention`) sobre el notebook y el builder: sin coincidencias. `nbformat.validate`: OK. El notebook de `src/` queda sin outputs.

## Problemas encontrados y corregidos

1. **El kernel `python3` no era el env `tp1`.** Existe un kernelspec de usuario en `C:\Users\facum\AppData\Roaming\jupyter\kernels\python3` que apunta a `C:\Python314\python.exe` (sin torchaudio), así que la primera corrida falló en `import torchaudio`. Sin tocar ese kernelspec global, se creó un kernelspec `python3` que apunta a `E:/Programs/miniconda/envs/tp1/python.exe` en `C:/Users/facum/tp1_cache/jupyter/kernels/python3/` y se antepuso con `JUPYTER_PATH=C:/Users/facum/tp1_cache/jupyter` (el comando de `CLAUDE.md` necesita esa variable extra en esta máquina). Las otras pistas pueden reutilizar el mismo `JUPYTER_PATH`.
2. **Formato de la tabla de parámetros** (floats por NaN al mezclar CNN y Transformers): se separó en dos tablas (total por modelo y desglose de A/B).
3. **Redacción de conclusiones** cuando el delta es negativo ("a favor de la CNN" con signo negativo) y de la brecha train-val: ahora el texto depende del signo.
4. Folds impresos como `np.int64(...)`: se convierten a `int`.

## Notas

- En el smoke las métricas no son interpretables (30 clips, 1 época); el notebook lo avisa en las conclusiones.
- No se instalaron paquetes en el env `tp1`.
