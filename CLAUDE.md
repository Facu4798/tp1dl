# CLAUDE.md

Este archivo lo lee Claude Code al inicio de cada sesión en este repo.

## Project context

- Plan: `PLAN.md` (en la raíz). Cada tarea lista qué archivos de `prompts/` y `context/` leer; leé solo esos.
- Enunciado original: `CONSIGNA.md`. Historial de decisiones del usuario: `QUESTIONS.md` (no se borra).
- Material de la cátedra: `context/` (PDFs y notebooks ya extraídos de los `.7z`). Los bloques del Transformer desde cero y el patrón HF Trainer + PEFT salen de ahí; seguí ese estilo.
- Todo el contenido de los notebooks (markdown, comentarios, prints) va en **español**.

## Trabajo en paralelo

- Las tareas 2–4 (Ej. 1), 5–7 (Ej. 2) y 8–10 (Ej. 3) son pistas independientes y las implementa un subagente por pista. Cada pista escribe **solo** sus propios archivos (`src/_build/build_ej{N}.py`, su notebook en `src/`, `outputs/smoke/ej{N}*`, `docs/decisiones_ej{N}.md`). No edites `CLAUDE.md`, `PLAN.md`, `prompts/`, `QUESTIONS.md` ni archivos de otra pista.
- El entorno conda `tp1` ya existe (Task 1): no lo recrees ni instales paquetes en él salvo que falte algo imprescindible, y en ese caso anotalo en tu `*_smoke_status.md`.

## Conventions

- Todo lo que genera correr código (notebooks ejecutados, CSVs, imágenes, JSON, checkpoints) va en `outputs/`, no junto al código ni en `docs/`. Datasets y checkpoints del smoke test van a `TP1_LOCAL_DIR=C:/Users/facum/tp1_cache` (fuera de OneDrive).
- Notebooks entregables: `src/Adell_Giannice_Mazzola_Ej1_ESC50.ipynb`, `src/Adell_Giannice_Mazzola_Ej2_NER_CoNLL.ipynb`, `src/Adell_Giannice_Mazzola_Ej3_Swin_EuroSAT.ipynb`. Se generan con `src/_build/build_ej{N}.py` (nbformat). Editá el builder y regenerá; no edites el JSON a mano. Los notebooks de `src/` quedan **sin outputs**.
- Celda de configuración común a los tres notebooks, en este orden:
  1. `IN_COLAB = 'google.colab' in sys.modules`; `%pip install` y `drive.mount` solo si `IN_COLAB`.
  2. `SMOKE_TEST = os.environ.get('TP1_SMOKE_TEST') == '1'`.
  3. `BASE = '/content/drive/MyDrive/Colab_Notebooks/TP1_Adell_Giannice_Mazzola/ej{N}'` en Colab; si no, `os.environ.get('TP1_LOCAL_DIR', './tp1_local') + '/ej{N}'`.
  4. `SEED = 42` (random, numpy, torch, y `transformers.set_seed` en Ej. 2/3); `device` agnóstico (cuda/cpu).
  5. Impresión de versiones de librerías y de un dict `HPARAMS` con los hiperparámetros.
- **Resultados nunca en markdown.** El markdown solo lleva teoría y metodología. Conclusiones = celdas de código que imprimen f-strings calculadas sobre las métricas reales. Tablas con pandas; gráficos con matplotlib.
- Ej. 1: prohibido `nn.Transformer`, `nn.TransformerEncoder`, `nn.TransformerEncoderLayer` y `nn.MultiheadAttention`.
- Ej. 2/3: fp16 solo si `torch.cuda.is_available()`; métricas de memoria "N/A" en CPU.
- Estilo de código: comentarios breves y útiles como en la cátedra, sin comentarios redundantes (al final se pasa la skill `code-style`).

## How to verify a task is done

- Smoke test de un notebook (desde la raíz del proyecto, en Git Bash):
  `JUPYTER_PATH=E:/Programs/miniconda/envs/tp1/share/jupyter TP1_SMOKE_TEST=1 TP1_LOCAL_DIR=C:/Users/facum/tp1_cache E:/Programs/miniconda/envs/tp1/python.exe -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 src/<notebook>.ipynb --output-dir outputs/smoke --output ej{N}_smoke.ipynb`
  (`JUPYTER_PATH` es necesario porque el kernelspec `python3` del usuario apunta a Python 3.14, no al env `tp1`.)
- Validación de estructura: `E:/Programs/miniconda/envs/tp1/python.exe -c "import nbformat,sys; nbformat.validate(nbformat.read(sys.argv[1], 4))" src/<notebook>.ipynb`

## Explicitly out of scope

- QLoRA / LLM en 4 bits / gradient accumulation; 5-fold CV en ESC-50; modelos preentrenados en el Ej. 1; un notebook único combinado.
- Correr los notebooks completos con GPU (lo hacen los alumnos en Colab), armar el zip, push al Hub y W&B.
