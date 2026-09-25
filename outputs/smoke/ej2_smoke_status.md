# Smoke test — Ej. 2 (NER DistilBERT / CoNLL-2003)

- **Resultado:** PASS. `nbconvert` salió con exit code 0 y ninguna celda tiene outputs de error.
- **Tiempo:** unos 36 s de punta a punta en CPU con el modelo ya en caché. La primera corrida, que incluye la descarga de `distilbert-base-cased` (263 MB), tardó unos 64 s.
- **Notebook ejecutado:** `outputs/smoke/ej2_smoke.ipynb`
- **Comando** (desde la raíz, Git Bash):
  ```
  JUPYTER_PATH=E:/Programs/miniconda/envs/tp1/share/jupyter TP1_SMOKE_TEST=1 TP1_LOCAL_DIR=C:/Users/facum/tp1_cache \
  E:/Programs/miniconda/envs/tp1/python.exe -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1800 src/Adell_Giannice_Mazzola_Ej2_NER_CoNLL.ipynb --output-dir outputs/smoke --output ej2_smoke.ipynb
  ```

## Verificaciones

- El dataset carga con `load_dataset('eriktks/conll2003', revision='refs/convert/parquet')`. Los label names salen de las **features** del dataset (`Etiquetas (features del dataset): ['O', 'B-PER', ...]`); el fallback queda en el código.
- `print_trainable_parameters()` de LoRA: `trainable params: 154,377 || all params: 65,352,210 || trainable%: 0.2362`. Coincide con la cuenta teórica r·(d+k)·6 capas·2 matrices + classifier = 154,377, que el notebook verifica.
- Probe: 6,921 entrenables (solo `classifier`). Full: 65,197,833.
- La tabla comparativa muestra F1, accuracy, deltas vs. probe, parámetros, memoria ("N/A" en CPU), checkpoint (full 0.261 GB vs. LoRA 0.63 MB) y tiempo. También se generan los gráficos de barras, el reporte por entidad, la inferencia (con el adaptador cargado desde disco vía `PeftModel.from_pretrained`) y las conclusiones impresas.
- `src/Adell_Giannice_Mazzola_Ej2_NER_CoNLL.ipynb` queda válido (`nbformat.validate`), sin outputs y con `SMOKE_TEST` controlado por la variable de entorno (default `False`).
- Las métricas del smoke (64 oraciones, 1 época) no tienen significado; solo prueban que el código corre.

## Errores encontrados y corregidos

1. **Kernel equivocado:** el kernelspec `python3` a nivel usuario (`%APPDATA%\jupyter\kernels\python3`) apunta a otro Python (3.14) que no tiene `datasets`, y fallaba con `ModuleNotFoundError: datasets`. Se resolvió sin tocar configuración: `JUPYTER_PATH=E:/Programs/miniconda/envs/tp1/share/jupyter` hace que `python3` resuelva al kernelspec del env `tp1`. **El comando de `CLAUDE.md` necesita este prefijo en esta máquina** (probablemente afecta también a los Ej. 1 y 3).
2. **datasets 5.x:** `ds['train']['tokens']` devuelve un objeto `Column`, no una `list`, y el tokenizer lo rechazaba (`ValueError: text input must be of type str...`). Se corrigió envolviéndolo en `list(...)`.
3. **transformers 5.x:** `warmup_ratio` ya no existe en `TrainingArguments`, así que se usa `warmup_steps=0.1` (un float < 1 se interpreta como proporción). También se usan `processing_class=` y `eval_strategy=`. Esto se detectó antes de correr, inspeccionando la versión instalada.
4. Algunas conclusiones afirmaban cosas de forma incondicional. Se reescribieron para que dependan de los datos: la referencia "predecir siempre O", el ahorro de tiempo vs. parámetros y la brecha por entidad en valor absoluto.

## Advertencias no bloqueantes

- `huggingface_hub` avisa que no hay symlinks en Windows (solo afecta el tamaño de la caché).
- `pin_memory` sin acelerador y el LOAD REPORT de transformers (la cabeza `classifier` se inicializa de cero, lo esperado).
