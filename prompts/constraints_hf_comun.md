## Constraints — comunes a los Ejercicios 2 y 3 (Hugging Face, LoRA vs. full)

- Stack de la cátedra: HF `transformers` (`Trainer`, `TrainingArguments`), `datasets`, `peft` (`LoraConfig`, `get_peft_model`, `TaskType`, `print_trainable_parameters`), `evaluate`. Instalar como en NB2 y desinstalar `torchao`.
- **Precisión mixta** (fp16 en T4, solo si hay CUDA) y **checkpoints en Google Drive** para reanudar si la sesión se corta (`save_strategy='epoch'`, `resume_from_checkpoint` cuando existan checkpoints previos en la carpeta de salida).
- **Batch size 16–32.**
- **Mismas condiciones** para full y LoRA: mismos datos, épocas, batch, scheduler, semilla, max length/resolución; solo cambia el LR (LoRA usa un LR mayor, como en la cátedra) y qué parámetros se entrenan.
- **Épocas**: full y LoRA **3 épocas** con evaluación por época y `load_best_model_at_end=True` sobre la métrica de validación (early stopping).
- **LoRA** sobre módulos de **atención** del modelo.
- **Antes del fine-tuning** = **linear probe de 1 época**: backbone congelado, solo se entrena la cabeza; se reporta su métrica en test y se calculan los **deltas** full−probe y LoRA−probe.
- **Reportar**: métrica de la tarea antes/después, parámetros entrenables (`print_trainable_parameters` para LoRA y conteo equivalente para full/probe), **memoria de GPU** (pico con `torch.cuda.max_memory_allocated`, reseteando con `torch.cuda.reset_peak_memory_stats` antes de cada entrenamiento; si no hay CUDA, reportar N/A), **tamaño del checkpoint** (full en GB vs. adaptador en MB), tiempo de entrenamiento.
- No hay LLM en 4 bits en estos ejercicios: no aplica micro-batch 1–2 ni gradient accumulation.
