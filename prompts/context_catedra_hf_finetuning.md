## Context — material de cátedra para los Ej. 2 y 3 (Hugging Face + PEFT)

Los `.7z` ya están extraídos en carpetas homónimas dentro de `context/`.

- `context/Notebooks_HF_y_FT/NB2_Fine_tuning_completo_vs_eficiente.ipynb` (patrón principal a seguir):
  - `%pip install -q -U transformers datasets accelerate peft evaluate` y luego `%pip uninstall -q -y torchao` (la versión de Colab rompe a peft).
  - Drive con fallback local: `try: drive.mount('/content/drive'); BASE='/content/drive/MyDrive/Colab_Notebooks/...' except Exception: BASE='./...'`.
  - `AutoTokenizer`, `AutoModelFor...`, `TrainingArguments(output_dir, per_device_train_batch_size=16, num_train_epochs, learning_rate=2e-5 (full) / 1e-3 (LoRA), eval_strategy='epoch', report_to='none', fp16=True)`, `Trainer(...)`, `trainer.evaluate()`.
  - `LoraConfig(task_type=TaskType.XXX, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q_lin','v_lin'])` para DistilBERT; `get_peft_model`; `model.print_trainable_parameters()`.
  - Guardado: full con `trainer.save_model(...)`, LoRA con `model.save_pretrained(...)` (solo el adaptador); `dir_size_mb(path)` con `os.walk`; `n_trainable(m)`; tabla comparativa con pandas.
- `context/Notebooks_HF_y_FT/NB1_Ecosistema_y_Arquitecturas.ipynb`: `load_dataset` + `.map(batched=True)`, AutoModel vs AutoModelForXxx.
- `context/hugging face y fine tuning eficiente.pdf` (Unidad IV): encoder-only con cabeza sobre cada token para NER; full FT vs PEFT (memoria: pesos + gradientes + 2 momentos de Adam); LoRA W = W0 + (α/r)·BA, B=0 al inicio, r 4–16, parámetros por capa r·(d+k), inyectado en proyecciones de atención q/v; LR de LoRA mayor que full; fp16/bf16, warmup + scheduler coseno, evaluación periódica sobre validación; `save_pretrained` guarda solo el adaptador (MB); reproducibilidad: fijar versiones, semilla, registrar hiperparámetros.
- `context/VIT.pdf` (Unidad III): Swin — atención en ventanas locales (W-MSA) y desplazadas (SW-MSA), costo lineal, patch merging jerárquico (4 etapas), sesgo posicional relativo; transferencia reemplazando la cabeza; linear probe = encoder congelado + clasificador lineal (mide la calidad intrínseca de las features).
