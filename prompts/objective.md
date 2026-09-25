## Objective

Resolver los tres ejercicios de `CONSIGNA.md` en **tres notebooks** (uno por ejercicio):

1. **Ej. 1 — ESC-50** (Piczak 2015): cargar el dataset y representar las señales en tiempo y en frecuencia (mel-spectrogram), describiendo los parámetros de ventana y el trade-off tiempo-frecuencia; implementar y entrenar una CNN 2D sobre el mel-espectrograma y dos configuraciones (A y B) de un mismo Transformer encoder **desde cero** en PyTorch (patching estilo AST, positional encoding, multi-head attention, FFN, layer normalization); comparar los tres en test (accuracy, F1-macro, matriz de confusión), reportando número de parámetros y tiempo de entrenamiento; visualizar los pesos de atención de la Config B sobre el espectrograma.
2. **Ej. 2 — NER con DistilBERT (`distilbert-base-cased`) sobre `eriktks/conll2003`**: cargar y tokenizar; fine-tuning con LoRA (`get_peft_model`, target modules de atención) y, bajo las mismas condiciones, full fine-tuning; comparar métrica de la tarea (F1/accuracy), parámetros entrenables (`print_trainable_parameters`), memoria de GPU y tamaño del checkpoint (full en GB vs. adaptador en MB).
3. **Ej. 3 — Swin-Tiny (`microsoft/swin-tiny-patch4-window7-224`) sobre EuroSAT**: cargar con su image processor y augmentation básica; LoRA y full fine-tuning bajo las mismas condiciones; comparar accuracy, parámetros entrenables (`print_trainable_parameters`) y tamaño del checkpoint (full en GB vs. adaptador en MB), más memoria.

En los Ej. 2 y 3 se reporta la métrica **antes** del fine-tuning (linear probe) y **después**, para leer la mejora como un delta.
