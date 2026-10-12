
# INSTRUCCION GENERAL PARA CLAUDE CODE
Leer la siguiente consigna y ejecutarla utilizando la skill de `proyect_planner`. verificar luego el codigo generado utilizando la skill de `code-style`. El output debe ser una notebook que resuelva los tres ejercicios. Preguntar que libreria deberíamos utilizar. no hardcodear resultados en los markdowns. El notebook se va a ejecutar en colab con GPU.

---

# Grupo: Matías Adell, Agustín Giannice, Facundo Mazzola 

## Ejercicio 1 — Señales: Clasificación de sonido ambiental  [CNN 2D vs. Transformer A/B] 
**Paper de referencia:** Piczak (2015). ESC-50. 
****Dataset:**** ESC-50 (GitHub karolpiczak/
ESC-50). 

### Consignas: 
1. Cargar el dataset y representar las señales en tiempo y frecuencia (mel-spectrogram); describir los 
parámetros de ventana y el trade-off tiempo-frecuencia. 
2. Implementar tres modelos: una CNN 2D sobre el mel-espectrograma y dos configuraciones (A y B, ver 
NOTAS) de un mismo Transformer encoder DESDE CERO en PyTorch — tokenizar el espectrograma en 
parches (patching, estilo AST), positional encoding, multi-head attention, FFN y layer normalization. 
3. Entrenar los tres y comparar en test (accuracy, F1-macro, matriz de confusión), reportando número de 
parámetros y tiempo de entrenamiento; visualizar los pesos de atención de la Config B sobre la 
señal/espectrograma. 

## Ejercicio 2 — NLP con Transformers (Hugging Face): NER  [LoRA vs. full fine-tuning] 
**Modelo:** DistilBERT (distilbert-base-cased) 
**Dataset:** eriktks/conll2003 

### Consignas: 
1. Cargar eriktks/conll2003 y tokenizar (tarea: NER). 
2. Hacer fine-tuning de DistilBERT (distilbert-base-cased) con LoRA (get_peft_model, target modules de 
atención) y, bajo las mismas condiciones, full fine-tuning. 
3. Comparar full vs. LoRA: métrica de la tarea (F1/accuracy), parámetros entrenables 
(print_trainable_parameters), memoria de GPU y tamaño del checkpoint (full en GB vs. adaptador en 
MB). 

## Ejercicio 3 — Vision Transformers (Hugging Face)  [LoRA vs. full fine-tuning] 
**Modelo:** Swin-Tiny (microsoft/swin-tiny-patch4-window7-224) 
**Dataset:** EuroSAT 

### Consignas: 
1. Cargar EuroSAT con su image processor y augmentation básica. 
2. Hacer fine-tuning de Swin-Tiny (microsoft/swin-tiny-patch4-window7-224) con LoRA (get_peft_model, 
target modules de atención) y, bajo las mismas condiciones, full fine-tuning. 
3. Comparar full vs. LoRA: accuracy, parámetros entrenables (print_trainable_parameters) y tamaño del 
checkpoint (full en GB vs. adaptador en MB). 


# PRESENTACIÓN Y ENTREGA 
Cada grupo debe entregar un archivo comprimido con los Notebooks desarrollados. Los nombres del archivo 
comprimido y de los notebooks deben contener los apellidos de los integrantes del grupo en orden alfabético. 
Los notebooks deben incluir todas las celdas ejecutadas con resultados visibles y comentarios con el análisis. 

# FECHA DE ENTREGA Y PRESENTACIÓN 
La fecha límite de entrega es el 13 de octubre de 2026, hasta las 15:30. Ese día se realizará también la 
presentación de los trabajos durante el horario de la clase. 
NOTAS 
## Ejecución en Google Colab.
Todo debería correr en Colab con GPU T4 (16 GB) y RAM estándar (sin RAM extendida): 

- Precisión y guardado: para los ejercicios 2 y 3 usar precisión mixta (fp16 o bf16) y guardar los 
checkpoints en Google Drive (carpeta Colab_Notebooks) para reanudar si la sesión se corta; para el 
Ejercicio 1 fp32 está bien. 
- Épocas (orden de magnitud): ~20–50 en el Ejercicio 1 (modelos chicos, desde cero); 1–3 en los 
ejercicios 2 y 3 (fine-tuning). 
- Batch size (ejercicios 2 y 3): 16–32 en los encoders y en ViT/Swin/DeiT. Para el LLM en 4 bits, micro
batch 1–2 (para que entre en la T4) con gradient accumulation de 4–8 pasos (la acumulación recupera un 
lote lo bastante grande para entrenar estable sin usar más memoria) → lote efectivo 8–16. 
- Datasets: si son pequeños (Flowers-102, Beans, Caltech-101, DTD, STL-10, Oxford-IIIT Pet, ESC-50, 
UCI-HAR, FordA), usarlos completos; si son grandes, tomar un subconjunto fijo de entrenamiento: 30k 
ejemplos en clasificación de texto (AG News, SNLI), 3k diálogos en resumen (DialogSum, SAMSum) y 
10k imágenes en el Ejercicio 3 con full fine-tuning de ViT/Swin/DeiT (EuroSAT). 

## Transformer del Ejercicio 1.
El encoder Transformer debe implementarse desde cero en PyTorch (No debe 
usarse torch.nn.Transformer (ni torch.nn.TransformerEncoder, torch.nn.TransformerEncoderLayer o 
torch.nn.MultiheadAttention)): la atención escalada, las cabezas múltiples, el feed-forward y la normalización 
se programan explícitamente. Pueden partir de los notebooks de implementación del Transformer desde cero 
(prácticas de la Unidad II) y adaptar la entrada según el ejercicio: en señales 1D, segmentar la señal en 
ventanas; en audio, partir el espectrograma en parches. En ambos casos, proyectar cada token linealmente a 
d_model, sumar la codificación posicional y pasarlo por los bloques de atención. Se entrenan y comparan tres 
modelos: la CNN (1D o 2D según el ejercicio) y dos configuraciones del mismo encoder (mismo tamaño), que 
difieren en la tokenización y la regularización: 
- Config A: tokenización en secuencias cortas (~8–16 tokens), 2 capas, d_model 64, 2 cabezas, sin 
dropout, post-norm. 
- Config B: tokenización en secuencias largas (~50–200 tokens; elegir el tamaño según la longitud de la 
señal o la resolución del espectrograma), 2–3 capas, d_model 64–128, 2–4 cabezas, dropout 0.1–0.2, pre
norm con warmup. 

Reportar, para los tres, exactitud, número de parámetros y tiempo de entrenamiento. 

**Entrenamiento.** Entrenar hasta que la métrica de validación se estabilice (early stopping) y mostrar la curva 
de entrenamiento/validación. En los ejercicios 2 y 3, reportar la métrica antes del fine-tuning (modelo base o 
zero-shot) y después, para leer la mejora como un delta, además de los parámetros entrenables, el tamaño del 
checkpoint y la memoria. 

**Reproducibilidad.** Fijar la semilla aleatoria, registrar las versiones de las librerías y reportar los 
hiperparámetros utilizados. 
REFERENCIAS 

# Referencias comunes a todos los trabajos. 

Los datasets específicos del Ejercicio 1 se citan dentro de cada trabajo.


## Métodos PEFT 
- Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. 
- Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314. 

## Arquitecturas 
- Vaswani, A. et al. (2017). Attention Is All You Need. arXiv:1706.03762. 
- Dosovitskiy, A. et al. (2021). An Image Is Worth 16x16 Words: Transformers for Image Recognition at 
Scale (ViT). arXiv:2010.11929. 
- Liu, Z. et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. 
arXiv:2103.14030. 
- Touvron, H. et al. (2021). Training Data-Efficient Image Transformers & Distillation through Attention 
(DeiT). arXiv:2012.12877. 
- Gong, Y. et al. (2021). AST: Audio Spectrogram Transformer. arXiv:2104.01778. 
- Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108. 
- Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115. 
- Zhang, P. et al. (2024). TinyLlama: An Open-Source Small Language Model. arXiv:2401.02385. 

## Otras referencias de interés 
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language 
Understanding. arXiv:1810.04805.