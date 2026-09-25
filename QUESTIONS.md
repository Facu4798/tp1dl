# QUESTIONS

## Ronda 1

Q: ¿Qué librería usamos para audio en el Ejercicio 1 (carga + mel-spectrogram)?
A: torchaudio.

Q: ¿Qué librería/loop de entrenamiento usamos en los Ejercicios 2 y 3 (LoRA vs full)?
A: Usar lo que se usa en el material de la cátedra (presentaciones y notebooks en context/). Revisado: NB2_Fine_tuning_completo_vs_eficiente usa HF `Trainer` + `TrainingArguments` (fp16=True, eval_strategy='epoch', report_to='none'), `peft` (`LoraConfig`, `get_peft_model`, `TaskType`, `print_trainable_parameters`), `evaluate` para métricas, `datasets.load_dataset` + `.map()`, guardado en Drive en `/content/drive/MyDrive/Colab_Notebooks/...`, tamaño de checkpoint con `os.walk`, full con `trainer.save_model`, LoRA con `model.save_pretrained` (solo adaptador). Instalación: `%pip install -q -U transformers datasets accelerate peft evaluate` (+ quitar torchao).

Q: ¿Un solo notebook con los 3 ejercicios o un notebook por ejercicio?
A: Tres notebooks (uno por ejercicio).

Q: ¿Tienen material de la cátedra que quieras que siga como base?
A: Sí, está en context/: leer todos los PDFs y extraer/leer los notebooks de los .7z. Revisado:
- Notebooks_Transformer (Practica1-4): bloques desde cero `scaled_dot_product_attention`, `MultiHeadAttention` (Wq/Wk/Wv/Wo, split en cabezas, devuelve attn), `PositionwiseFFN` (Linear-ReLU-Linear), `PositionalEncoding` sinusoidal, `EncoderLayer` post-norm con dropout; Practica4 usa proyección lineal de valores continuos a d_model, pooling medio y visualización de atención promedio sobre posiciones de entrada.
- Practica_Filtrado_Senales: estilo con SEED=42, helpers de plot tiempo/espectro, curvas train/val, tablas pandas comparativas, CNN con Conv-BN-ReLU.
- señales.pdf: STFT (ventana N, hop H, Hann, trade-off tiempo-frecuencia, Gabor), mel scale, mel-spectrogram en dB, torchaudio MelSpectrogram, SpecAugment (FrequencyMasking/TimeMasking); receta ESC-50: resamplear a 22050 Hz, n_mels=128, dB, SpecAugment en train; CNN 2D simple ~70% acc como referencia.
- transformers.pdf: atención escalada, MHA, encoder (residual+LayerNorm), PE sinusoidal/aprendido.
- VIT.pdf: patch embedding, class token, PE aprendido, pre-norm (LN→MSA→residual, LN→MLP GELU→residual), Swin (W-MSA/SW-MSA, patch merging, sesgo posicional relativo).
- hugging face y fine tuning eficiente.pdf: LoRA (r 4–16, alpha, dropout, target q/v de atención), LR de LoRA mayor que full, fp16/bf16, warmup + cosine, eval periódica, save_pretrained solo adaptador, reproducibilidad (fijar versiones, semilla, hiperparámetros).

## Ronda 2

Q: ESC-50 trae 5 folds predefinidos. ¿Cómo particionamos train/val/test en el Ej. 1?
A: Folds 1-3 train / fold 4 validación / fold 5 test. Documentar toda la justificación en un documento aparte.

Q: La consigna pide la métrica ANTES del fine-tuning (Ej. 2 y 3). ¿Cómo la medimos?
A: Linear probe (backbone congelado, solo se entrena la cabeza) como "antes".

Q: Como no se pueden hardcodear resultados en markdown, ¿cómo manejamos las celdas de análisis/conclusiones?
A: Texto generado por código: celdas que imprimen conclusiones con f-strings a partir de las métricas reales (quién ganó, deltas, ratios). Markdown solo con teoría/metodología.

Q: EuroSAT (Ej. 3): la consigna dice 10k imágenes de train para full FT. ¿Qué aplicamos a LoRA?
A: Mismo subconjunto fijo de 10k (estratificado, semilla fija) para full y LoRA — mismas condiciones. Val/test de los splits oficiales.

## Ronda 3

Q: Regularización en Ej. 1: ¿dónde aplicamos SpecAugment?
A: En la CNN y en la Config B. La Config A queda sin ninguna regularización (fiel a "sin dropout"); el contraste A/B se lee como tokenización + regularización.

Q: Tokenización de la Config A (~8-16 tokens) sobre un mel de 128 bins x ~216 frames. ¿Qué forma de parche?
A: Franjas temporales: ~12 franjas de banda completa (128 mel x ~18 frames); cada token = un segmento de tiempo con todo su espectro. Contrasta con los parches 2D (estilo AST) de B.

Q: El documento aparte de justificación, ¿solo para el split o para todas las decisiones?
A: Todas las decisiones de diseño de los 3 ejercicios: docs/decisiones_de_diseno.md (split, STFT/mel, tokenización A/B, hiperparámetros, LoRA r/alpha, linear probe, subset 10k, etc.) con su justificación.

Q: ¿Hago una prueba de humo local (CPU, subset diminuto, 1 época) de cada notebook antes de entregarlos?
A: Sí. Flag SMOKE_TEST que achica datos/épocas para verificar que corre de punta a punta; los resultados reales se generan en Colab.

## Ronda 4

Q: Python local es 3.14; ¿puedo crear un entorno conda aparte con Python 3.11 en E:/Programs/miniconda para el smoke test?
A: Sí, env conda 'tp1' (python 3.11 + torch CPU, torchaudio, transformers, datasets, peft, evaluate, seqeval, nbconvert).

Q: Épocas de fine-tuning (Ej. 2 y 3) y del linear probe:
A: Full y LoRA 3 épocas con load_best_model_at_end sobre validación (early stopping); linear probe 1 época como "antes".

Q: Nombres de archivo y carpeta en Drive:
A: Adell_Giannice_Mazzola_Ej1_ESC50.ipynb, Adell_Giannice_Mazzola_Ej2_NER_CoNLL.ipynb, Adell_Giannice_Mazzola_Ej3_Swin_EuroSAT.ipynb; Drive: MyDrive/Colab_Notebooks/TP1_Adell_Giannice_Mazzola/ej{1,2,3}/; zip: Adell_Giannice_Mazzola_TP1.zip.

Q: (Mensaje del usuario, no pregunta) ¿Qué se borra en la limpieza final?
A: Además del documento de justificación, NO borrar QUESTIONS.md en esta ocasión (se borra solo PROMPT.md).

## Ronda 5 (aprobación del prompt expandido)

Q: ¿Apruebo el prompt expandido y paso a escribir PROMPT.md, prompts/, PLAN.md y CLAUDE.md?
A: Aprobado, con un cambio para ahorrar tokens: hacer el plan paralelizable por notebook y lanzar tres subagentes (uno por notebook) sin el contexto de esta conversación. Análisis aceptado con ajustes: (1) el env conda `tp1` y las convenciones comunes se crean antes y en serie; (2) cada agente escribe `docs/decisiones_ej{N}.md` y al final se unifican en `docs/decisiones_de_diseno.md`; (3) la pasada de `code-style`, la revisión cruzada y la limpieza se hacen al final en serie.
