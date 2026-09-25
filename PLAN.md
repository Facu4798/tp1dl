# Plan: TP1 Deep Learning II — Adell, Giannice, Mazzola

## Problem & goal

Resolver los tres ejercicios de `CONSIGNA.md` en tres notebooks de Colab (T4): ESC-50 con CNN 2D vs. Transformer desde cero (Config A/B), NER con DistilBERT (LoRA vs. full) y Swin-Tiny en EuroSAT (LoRA vs. full). "Hecho" = los tres notebooks corren de punta a punta en modo `SMOKE_TEST` en el env local `tp1`, en modo completo respetan la consigna, no tienen resultados hardcodeados en markdown y existe `docs/decisiones_de_diseno.md` con la justificación de todas las decisiones.

## Key decisions

- Tres notebooks independientes en `src/`; cada uno se genera desde un script builder (`src/_build/build_ej{N}.py`, con `nbformat`) para poder editarlo y regenerarlo sin tocar JSON a mano.
- `SMOKE_TEST = os.environ.get('TP1_SMOKE_TEST') == '1'` en la celda de configuración: default `False` en Colab; el smoke test local se activa por variable de entorno sin editar el notebook.
- Fuera de Colab, datos y checkpoints van a `TP1_LOCAL_DIR` (default `C:/Users/facum/tp1_cache` en el smoke test, **fuera de OneDrive**, para no sincronizar GBs); en Colab, a `MyDrive/Colab_Notebooks/TP1_Adell_Giannice_Mazzola/ej{N}/`.
- Ej. 1: torchaudio, 22050 Hz, n_mels=128 en dB; split folds 1–3/4/5; Config A = ~12 franjas temporales, post-norm, sin regularización; Config B = parches 2D AST, pre-norm + warmup + dropout + SpecAugment; CNN con SpecAugment; encoder escrito a mano, sin clases `nn.Transformer*` ni `nn.MultiheadAttention`.
- Ej. 2/3: HF `Trainer` + `peft` como en NB2 de la cátedra; fp16 si hay CUDA; 3 épocas + `load_best_model_at_end`; "antes" = linear probe de 1 época; LoRA en atención (`q_lin`/`v_lin`; `query`/`value`) con `modules_to_save=['classifier']`.
- CoNLL: `load_dataset("eriktks/conll2003", revision="refs/convert/parquet")`. EuroSAT: `timm/eurosat-rgb`, subset estratificado de 10k para full y LoRA.
- Conclusiones impresas desde código (f-strings sobre métricas reales); el markdown solo lleva teoría y metodología.
- **Paralelización**: la Fase 1 corre en serie; las tres pistas de la Fase 2 (Ej. 1, Ej. 2, Ej. 3) son independientes y las implementan tres subagentes en paralelo, cada uno solo con sus archivos; la Fase 3 corre en serie al final.

# Phase 1 — Preparación compartida (en serie)

## Task 1 — Entorno conda `tp1` para el smoke test
**Prompts:** constraints_verificacion.md
**Context:** none
**Objective:** Tener un entorno local con Python 3.11 y todas las dependencias CPU para ejecutar los notebooks en modo smoke.
**Description:** Crear `tp1` con `E:/Programs/miniconda/Scripts/conda.exe create -n tp1 python=3.11`; instalar torch/torchaudio/torchvision CPU (index `https://download.pytorch.org/whl/cpu`) y luego transformers, datasets, accelerate, peft, evaluate, seqeval, scikit-learn, pandas, matplotlib, jupyter, nbconvert, nbformat, ipykernel y soundfile. Registrar las versiones instaladas. Se hace una sola vez, antes de la Fase 2.
**Outputs:**
- `E:/Programs/miniconda/envs/tp1/` — created
- `outputs/env_tp1_versions.txt` — created (salida de `pip freeze` filtrada y la versión de Python)
**Definition of done:** `E:/Programs/miniconda/envs/tp1/python.exe -c "import torch, torchaudio, torchvision, transformers, datasets, peft, evaluate, seqeval, nbformat"` sale sin error.

# Phase 2 — Notebooks (tres pistas paralelas e independientes)

> Cada pista la ejecuta un subagente distinto. Una pista solo escribe en sus propios archivos: `src/_build/build_ej{N}.py`, `src/Adell_Giannice_Mazzola_Ej{N}_*.ipynb`, `outputs/smoke/ej{N}*` y `docs/decisiones_ej{N}.md`. Ninguna pista edita `CLAUDE.md`, `PLAN.md`, `prompts/` ni archivos de otra pista.

## Task 2 — [Pista Ej. 1] Construir el notebook ESC-50
**Prompts:** role.md, objective.md, constraints_plataforma.md, constraints_ej1.md, constraints_entregables.md, context_catedra_senales_transformer.md, context_datos.md, non_goals.md
**Context:** context/Notebooks_Transformer/Practica1_Transformer_Fechas.ipynb, context/Notebooks_Transformer/Practica4_Transformer_Series_Energia.ipynb, context/Practica_Filtrado_Senales/Practica_Filtrado_Senales.ipynb, context/señales.pdf (págs. 37–44, 58–65, 73–82, 90–93), context/VIT.pdf (págs. 9–14)
**Objective:** Generar `src/Adell_Giannice_Mazzola_Ej1_ESC50.ipynb` que resuelva las tres consignas del Ej. 1.
**Description:** Estructura sugerida:
0. Título, integrantes, índice.
1. Setup: detección de Colab, `%pip` solo en Colab, Drive con fallback a `TP1_LOCAL_DIR`, `SMOKE_TEST`, semilla, device, versiones.
2. Datos: descarga y extracción del zip de ESC-50 (idempotente), `meta/esc50.csv`, split por fold; en smoke, pocos clips por fold (p. ej. 2 por clase de 5 clases).
3. Representación: forma de onda en el tiempo de algunas clases, STFT lineal vs. mel en dB; markdown sobre n_fft, hop, ventana Hann, resoluciones Δf = fs/n_fft y Δt = hop/fs calculadas por código, y trade-off tiempo-frecuencia (Gabor); demo con dos tamaños de ventana.
4. Precomputar los mel en dB de todo el dataset en un tensor cacheado en disco, normalizado con media y desvío de train; SpecAugment con `torchaudio.transforms.FrequencyMasking`/`TimeMasking` solo en train para la CNN y B.
5. Modelos: CNN 2D (bloques Conv-BN-ReLU-MaxPool + GAP + dropout); bloques del Transformer desde cero adaptados de la cátedra (`scaled_dot_product_attention`, `MultiHeadAttention`, `PositionwiseFFN`, `PositionalEncoding` o PE aprendido, `EncoderLayer` con flag `pre_norm`); un único `SpectrogramTransformer(patch=(ph, pw), d_model, n_heads, n_layers, dropout, pre_norm)` con token CLS, proyección lineal de parches y opción de devolver la atención. Config A: patch=(128, 18) → 12 tokens, 2 capas, d=64, h=2, dropout=0, post-norm. Config B: parches 2D (p. ej. 16×16 → 8×13 = 104 tokens), 2 capas, d=64, h=4, dropout 0.1–0.2, pre-norm. Imprimir los parámetros de A y B para mostrar que son comparables.
6. Entrenamiento: loop propio, AdamW, early stopping con paciencia sobre accuracy de validación, máximo ~50 épocas (`SMOKE_TEST`: 1), warmup lineal + coseno para B, guardado del mejor estado, tiempo de pared medido; curvas train/val de los tres.
7. Evaluación en test: accuracy, F1-macro (sklearn), matrices de confusión de los tres (50×50, heatmap), tabla pandas con acc, F1, parámetros y tiempo.
8. Atención de B: atención CLS→parches de la última capa, promediada por cabezas (y rollout opcional), reshape a la grilla de parches, upsample al tamaño del espectrograma y superposición sobre ejemplos de test (acierto y error).
9. Conclusiones impresas por código a partir de las métricas: ranking, deltas A vs. B, eficiencia en parámetros y tiempo, comparación con la referencia de la cátedra (~70% para la CNN).
10. Referencias (Piczak 2015, Gong 2021 AST, Vaswani 2017, Dosovitskiy 2021, Park 2019 SpecAugment).
El markdown solo explica; no escribe ningún número de resultado. El builder genera el `.ipynb` sin outputs.
**Outputs:**
- `src/_build/build_ej1.py` — created
- `src/Adell_Giannice_Mazzola_Ej1_ESC50.ipynb` — created (generado por el builder)
**Definition of done:** El builder corre y genera un notebook válido (`nbformat.validate`). `grep -nE "nn\.(Transformer|TransformerEncoder|TransformerEncoderLayer|MultiheadAttention)"` sobre el notebook no devuelve nada. Cada ítem de `prompts/constraints_ej1.md` tiene su celda.

## Task 3 — [Pista Ej. 1] Smoke test del notebook ESC-50
**Prompts:** constraints_verificacion.md, constraints_plataforma.md
**Context:** none
**Objective:** Verificar que el notebook del Ej. 1 corre de punta a punta en CPU.
**Description:** Con `TP1_SMOKE_TEST=1` y `TP1_LOCAL_DIR=C:/Users/facum/tp1_cache`, ejecutar `E:/Programs/miniconda/envs/tp1/python.exe -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 src/Adell_Giannice_Mazzola_Ej1_ESC50.ipynb --output-dir outputs/smoke --output ej1_smoke.ipynb`. Si falla, corregir en el builder, regenerar y repetir. No dejar outputs en el notebook de `src/`.
**Outputs:**
- `outputs/smoke/ej1_smoke.ipynb` — created (notebook ejecutado en modo smoke)
- `outputs/smoke/ej1_smoke_status.md` — created (pass/fail, tiempo, errores corregidos)
- `src/_build/build_ej1.py`, `src/Adell_Giannice_Mazzola_Ej1_ESC50.ipynb` — modified si hubo fixes
**Definition of done:** nbconvert termina con exit code 0 y el notebook ejecutado muestra la tabla comparativa, las matrices de confusión, la atención de B y las conclusiones impresas.

## Task 4 — [Pista Ej. 1] Justificación de decisiones del Ej. 1
**Prompts:** constraints_ej1.md, constraints_entregables.md, context_catedra_senales_transformer.md
**Context:** context/señales.pdf
**Objective:** Documentar y justificar cada decisión de diseño del Ej. 1.
**Description:** En español, una sección por decisión con qué se eligió, alternativas y por qué: split 1–3/4/5 (vs. 5-fold CV: costo en Colab; fold 4 para early stopping sin tocar test; folds oficiales para evitar fuga del mismo clip de origen); 22050 Hz y n_mels=128 (receta de la cátedra; Nyquist 11 kHz); n_fft, hop y ventana con sus resoluciones; normalización; SpecAugment solo en la CNN y B; tokenización A (franjas) vs. B (parches 2D) y cantidad de tokens; post-norm vs. pre-norm + warmup; d_model, cabezas y capas con el conteo de parámetros; CNN; optimizador, LR, early stopping, épocas; método de visualización de la atención; semilla.
**Outputs:**
- `docs/decisiones_ej1.md` — created
**Definition of done:** Cada hiperparámetro/decisión que aparece en el notebook del Ej. 1 tiene su entrada justificada en el documento.

## Task 5 — [Pista Ej. 2] Construir el notebook NER DistilBERT
**Prompts:** role.md, objective.md, constraints_plataforma.md, constraints_hf_comun.md, constraints_ej2.md, constraints_entregables.md, context_catedra_hf_finetuning.md, context_datos.md, non_goals.md
**Context:** context/Notebooks_HF_y_FT/NB2_Fine_tuning_completo_vs_eficiente.ipynb, context/Notebooks_HF_y_FT/NB1_Ecosistema_y_Arquitecturas.ipynb, context/hugging face y fine tuning eficiente.pdf (págs. 16–18, 28–39, 58, 60–64)
**Objective:** Generar `src/Adell_Giannice_Mazzola_Ej2_NER_CoNLL.ipynb` que resuelva las tres consignas del Ej. 2.
**Description:** Estructura sugerida:
0. Título, integrantes, índice.
1. Setup: `%pip` en Colab y uninstall `torchao`; Drive con fallback; `SMOKE_TEST`; `set_seed`; versiones; dict de hiperparámetros impreso.
2. Datos: `load_dataset("eriktks/conll2003", revision="refs/convert/parquet")`, label names con fallback, ejemplos y distribución de entidades; en smoke, `select` de ~64/32/32 ejemplos.
3. Tokenización: `is_split_into_words=True`, alineación de labels (primer subtoken, resto -100), ejemplo visual token↔label, `DataCollatorForTokenClassification`.
4. Métricas: `evaluate.load('seqeval')` → precision, recall, F1, accuracy; helpers `n_trainable`, `dir_size`, medición de memoria (`reset_peak_memory_stats` / `max_memory_allocated`, N/A sin CUDA) y tiempo.
5. Función común `run_experiment(mode, lr, epochs)` que arma el modelo (`probe`: congela `distilbert.*`; `lora`: `get_peft_model` + `LoraConfig(TaskType.TOKEN_CLS, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q_lin','v_lin'], modules_to_save=['classifier'])` + `print_trainable_parameters()`; `full`: todo entrenable), `TrainingArguments` idénticos salvo el LR (batch 16–32, fp16 si hay CUDA, `eval_strategy`/`save_strategy='epoch'`, `load_best_model_at_end=True`, `metric_for_best_model='f1'`, `save_total_limit`, `report_to='none'`, warmup), reanudación desde el último checkpoint si existe, evaluación en test, guardado (full con `save_model`, LoRA con `save_pretrained` solo adaptador) y tamaño en disco.
6. Correr probe (1 época), full (3) y LoRA (3); curvas train/val desde `trainer.state.log_history`.
7. Tabla comparativa: F1/acc antes (probe) y después, deltas, parámetros entrenables (abs y %), memoria pico, tamaño del checkpoint (full en GB, adaptador en MB), tiempo. Gráficos de barras.
8. Reporte por entidad (seqeval `classification_report`) de full y LoRA, e inferencia sobre una oración de ejemplo.
9. Conclusiones impresas por código (ratio de parámetros, ratio de tamaño, gap de F1 full vs. LoRA, mejora sobre el probe).
10. Referencias (Sanh 2019, Hu 2021, Tjong Kim Sang 2003, Devlin 2019).
**Outputs:**
- `src/_build/build_ej2.py` — created
- `src/Adell_Giannice_Mazzola_Ej2_NER_CoNLL.ipynb` — created
**Definition of done:** El builder genera un notebook válido (`nbformat.validate`). Probe, full y LoRA comparten `TrainingArguments` salvo LR y épocas del probe. Cada ítem de `constraints_ej2.md` y `constraints_hf_comun.md` tiene su celda.

## Task 6 — [Pista Ej. 2] Smoke test del notebook NER
**Prompts:** constraints_verificacion.md, constraints_plataforma.md
**Context:** none
**Objective:** Verificar que el notebook del Ej. 2 corre de punta a punta en CPU.
**Description:** Igual que la Task 3 pero con `src/Adell_Giannice_Mazzola_Ej2_NER_CoNLL.ipynb` → `outputs/smoke/ej2_smoke.ipynb`. Confirmar que el dataset carga vía `refs/convert/parquet` y que los label names salen de las features o del fallback.
**Outputs:**
- `outputs/smoke/ej2_smoke.ipynb` — created
- `outputs/smoke/ej2_smoke_status.md` — created
- `src/_build/build_ej2.py`, `src/Adell_Giannice_Mazzola_Ej2_NER_CoNLL.ipynb` — modified si hubo fixes
**Definition of done:** nbconvert termina con exit code 0; el notebook ejecutado muestra la salida de `print_trainable_parameters`, la tabla comparativa con tamaños de checkpoint y las conclusiones impresas.

## Task 7 — [Pista Ej. 2] Justificación de decisiones del Ej. 2
**Prompts:** constraints_hf_comun.md, constraints_ej2.md, constraints_entregables.md, context_catedra_hf_finetuning.md, context_datos.md
**Context:** context/hugging face y fine tuning eficiente.pdf
**Objective:** Documentar y justificar cada decisión de diseño del Ej. 2.
**Description:** Carga vía parquet (script no soportado en `datasets` ≥ 4); dataset completo; alineación de labels al primer subtoken; seqeval a nivel entidad; linear probe como "antes" (vs. cabeza aleatoria); LoRA r/alpha/dropout/targets y `modules_to_save`; LRs de full y LoRA; batch, épocas, warmup, fp16, `load_best_model_at_end`; medición de memoria y de tamaño; qué significa "mismas condiciones".
**Outputs:**
- `docs/decisiones_ej2.md` — created
**Definition of done:** Cada hiperparámetro/decisión del notebook del Ej. 2 está justificado en el documento.

## Task 8 — [Pista Ej. 3] Construir el notebook Swin EuroSAT
**Prompts:** role.md, objective.md, constraints_plataforma.md, constraints_hf_comun.md, constraints_ej3.md, constraints_entregables.md, context_catedra_hf_finetuning.md, context_datos.md, non_goals.md
**Context:** context/Notebooks_HF_y_FT/NB2_Fine_tuning_completo_vs_eficiente.ipynb, context/VIT.pdf (págs. 9–27, 44–45, 69), context/hugging face y fine tuning eficiente.pdf (págs. 30–39)
**Objective:** Generar `src/Adell_Giannice_Mazzola_Ej3_Swin_EuroSAT.ipynb` que resuelva las tres consignas del Ej. 3.
**Description:** Estructura sugerida:
0. Título, integrantes, índice.
1. Setup: igual que en el Ej. 2, con torchvision.
2. Datos: `load_dataset("timm/eurosat-rgb")`, clases, grilla de ejemplos por clase; subset estratificado de 10k de train con semilla fija (p. ej. `train_test_split(stratify_by_column='label', train_size=10000, seed=SEED)`; en smoke ~100/50/50); distribución de clases del subset.
3. Preprocesamiento: `AutoImageProcessor` (tamaño, mean y std leídos del processor), augmentation básica en train (RandomResizedCrop/flip horizontal y vertical/rotación leve) + Normalize, eval solo resize + normalize; `with_transform`; visualización de imágenes aumentadas.
4. Helpers compartidos (conteo de parámetros, tamaño de carpeta, memoria, tiempo, `compute_metrics` con accuracy y F1-macro).
5. `run_experiment(mode, lr, epochs)` análogo al del Ej. 2: `probe` congela `swin.*`; `lora` con `LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['query','value'], modules_to_save=['classifier'])` + `print_trainable_parameters()`; `full`. `remove_unused_columns=False`, collator de `pixel_values`/`labels`, fp16 si hay CUDA, checkpoints por época en Drive con reanudación, `load_best_model_at_end` por accuracy.
6. Correr probe (1 época), full (3) y LoRA (3); curvas.
7. Tabla comparativa: accuracy antes/después, deltas, parámetros entrenables, memoria pico, tamaño del checkpoint (GB vs. MB), tiempo; matriz de confusión del mejor modelo; ejemplos mal clasificados.
8. Conclusiones impresas por código.
9. Referencias (Liu 2021 Swin, Helber 2019 EuroSAT, Hu 2021 LoRA, Dosovitskiy 2021).
**Outputs:**
- `src/_build/build_ej3.py` — created
- `src/Adell_Giannice_Mazzola_Ej3_Swin_EuroSAT.ipynb` — created
**Definition of done:** El builder genera un notebook válido. El subset de 10k es el mismo objeto para full y LoRA. Cada ítem de `constraints_ej3.md` y `constraints_hf_comun.md` tiene su celda.

## Task 9 — [Pista Ej. 3] Smoke test del notebook Swin
**Prompts:** constraints_verificacion.md, constraints_plataforma.md
**Context:** none
**Objective:** Verificar que el notebook del Ej. 3 corre de punta a punta en CPU.
**Description:** Igual que la Task 3 pero con el notebook del Ej. 3 → `outputs/smoke/ej3_smoke.ipynb`.
**Outputs:**
- `outputs/smoke/ej3_smoke.ipynb` — created
- `outputs/smoke/ej3_smoke_status.md` — created
- `src/_build/build_ej3.py`, `src/Adell_Giannice_Mazzola_Ej3_Swin_EuroSAT.ipynb` — modified si hubo fixes
**Definition of done:** nbconvert termina con exit code 0; el notebook ejecutado muestra `print_trainable_parameters`, la tabla comparativa y las conclusiones impresas.

## Task 10 — [Pista Ej. 3] Justificación de decisiones del Ej. 3
**Prompts:** constraints_hf_comun.md, constraints_ej3.md, constraints_entregables.md, context_catedra_hf_finetuning.md, context_datos.md
**Context:** context/VIT.pdf
**Objective:** Documentar y justificar cada decisión de diseño del Ej. 3.
**Description:** Dataset `timm/eurosat-rgb` y sus splits; subset estratificado de 10k para ambos métodos; upscale de 64 a 224 con el image processor; augmentations elegidas (válidas para imágenes satelitales: flips verticales y rotaciones); linear probe; LoRA en `query`/`value` y cabeza en `modules_to_save`; LRs, batch, épocas, fp16; métricas; medición de memoria y de tamaño.
**Outputs:**
- `docs/decisiones_ej3.md` — created
**Definition of done:** Cada hiperparámetro/decisión del notebook del Ej. 3 está justificado en el documento.

# Phase 3 — Integración y cierre (en serie, después de las tres pistas)

## Task 11 — Documento unificado de decisiones
**Prompts:** constraints_entregables.md
**Context:** none
**Objective:** Unificar las justificaciones en un solo documento.
**Description:** Unir `docs/decisiones_ej{1,2,3}.md` en `docs/decisiones_de_diseno.md`, con una intro, una sección de decisiones transversales (tres notebooks, `SMOKE_TEST`, Drive, semilla, conclusiones por código, carga de datasets) y una sección por ejercicio. Borrar los parciales una vez unidos.
**Outputs:**
- `docs/decisiones_de_diseno.md` — created
- `docs/decisiones_ej1.md`, `docs/decisiones_ej2.md`, `docs/decisiones_ej3.md` — deleted
**Definition of done:** El documento unificado contiene las tres secciones completas más la sección transversal.

## Task 12 — Revisión cruzada y pasada de `code-style`
**Prompts:** definition_of_done.md, constraints_plataforma.md, constraints_verificacion.md
**Context:** none
**Objective:** Asegurar consistencia entre notebooks y aplicar la skill `code-style` al código Python.
**Description:** Aplicar la skill `code-style` sobre `src/_build/build_ej*.py` (el código de las celdas vive ahí) y regenerar los notebooks. Revisar la consistencia de la celda de setup, los nombres y el formato de tablas y conclusiones. Grep de números en markdown (ningún resultado hardcodeado) y grep de clases prohibidas en el Ej. 1. Si algo cambió, volver a correr los tres smoke tests.
**Outputs:**
- `src/_build/build_ej{1,2,3}.py` — modified
- `src/Adell_Giannice_Mazzola_Ej{1,2,3}_*.ipynb` — regenerated
- `outputs/smoke/review_status.md` — created (checks realizados y resultados)
**Definition of done:** Los tres smoke tests vuelven a pasar después de los cambios y los greps salen limpios.

## Task 13 — Limpieza del planner
**Prompts:** constraints_verificacion.md
**Context:** none
**Objective:** Dejar el proyecto en su estado final.
**Description:** Borrar `PROMPT.md`. **No** borrar `QUESTIONS.md` ni `prompts/`. Confirmar que los notebooks en `src/` quedan sin outputs y con `SMOKE_TEST` en `False` por defecto.
**Outputs:**
- `PROMPT.md` — deleted
**Definition of done:** `PROMPT.md` no existe; `QUESTIONS.md`, `prompts/`, `docs/decisiones_de_diseno.md` y los tres notebooks sí.
