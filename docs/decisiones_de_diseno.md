# Decisiones de diseño — TP1 Deep Learning II

**Grupo:** Matías Adell, Agustín Giannice, Facundo Mazzola

Este documento justifica las decisiones de diseño de los tres notebooks del TP1: qué se eligió, qué alternativas había y por qué. Los resultados numéricos **no** aparecen acá: los calcula e imprime cada notebook al ejecutarse en Colab.

## Índice

- [Decisiones transversales](#decisiones-transversales)
- [Ejercicio 1 — ESC-50: CNN 2D vs. Transformer A/B](#decisiones-de-diseño--ejercicio-1-esc-50-cnn-2d-vs-transformer-ab)
- [Ejercicio 2 — NER con DistilBERT: LoRA vs. full](#decisiones-de-diseño--ejercicio-2-ner-con-distilbert-sobre-conll-2003-lora-vs-full)
- [Ejercicio 3 — Swin-Tiny en EuroSAT: LoRA vs. full](#decisiones-de-diseño--ejercicio-3-swin-tiny-en-eurosat-lora-vs-full-fine-tuning)

## Decisiones transversales

### Tres notebooks independientes
Uno por ejercicio (`Adell_Giannice_Mazzola_Ej{1,2,3}_*.ipynb`). Cada ejercicio usa datasets, modelos y dependencias distintos. Separarlos limita el daño si se corta una sesión de Colab y permite correrlos en sesiones distintas dentro del límite de GPU gratuita.

### Stack de la cátedra
El Ej. 1 usa PyTorch + torchaudio, con los bloques del Transformer adaptados de las prácticas de la Unidad II. Los Ej. 2 y 3 usan HF `Trainer` + `peft` + `evaluate`, siguiendo el patrón de `NB2_Fine_tuning_completo_vs_eficiente`. Así el código es reconocible para la cátedra y comparable con lo visto en clase.

### Compatibilidad con versiones recientes
Colab instala con `pip -U`, así que el código apunta a las versiones actuales (verificadas localmente con transformers 5.17, datasets 5.0 y peft 0.21):
- `processing_class=` en `Trainer`, `eval_strategy=`, `warmup_steps` fraccional (ya no existe `warmup_ratio`).
- `list(...)` sobre las columnas de `datasets` 5, que devuelven `Column` en lugar de listas.
- Lectura de WAV con `soundfile`: el `torchaudio.load` reciente depende de torchcodec.
- En Swin, los nombres de las proyecciones de atención y la recarga de checkpoints se resuelven según la versión instalada (ver Ej. 3).

### Carga de datasets
- ESC-50: zip oficial del repo de GitHub, con descarga idempotente.
- CoNLL-2003: `eriktks/conll2003` solo trae un script de carga, que `datasets` ≥ 4 ya no ejecuta. Se usa la conversión oficial a parquet (`revision="refs/convert/parquet"`), que conserva los `ClassLabel`.
- EuroSAT: `timm/eurosat-rgb`, que trae splits oficiales train/validation/test.

### Reproducibilidad
- Semilla 42 en `random`, `numpy` y `torch`, y `transformers.set_seed` en los Ej. 2 y 3.
- Cada notebook imprime las versiones de las librerías y un dict `HPARAMS` con todos los hiperparámetros.

### Persistencia y reanudación
- En Colab, los artefactos van a `MyDrive/Colab_Notebooks/TP1_Adell_Giannice_Mazzola/ej{N}/`. Fuera de Colab van a `TP1_LOCAL_DIR`.
- Los Ej. 2 y 3 guardan checkpoints por época y reanudan con `resume_from_checkpoint`. Además cachean los resultados de cada experimento en JSON, para no repetir entrenamientos ya terminados si la sesión se corta.
- El Ej. 1 cachea los mel-espectrogramas y los mejores pesos de cada modelo.

### Resultados generados por código
La consigna prohíbe hardcodear resultados. El markdown solo contiene teoría y metodología; toda comparación o conclusión (quién gana, deltas, ratios) se imprime con f-strings calculadas sobre las métricas de esa ejecución.

### Modo `SMOKE_TEST`
`SMOKE_TEST = os.environ.get('TP1_SMOKE_TEST') == '1'`: vale `False` en Colab. Con la variable activa, los notebooks corren con subsets diminutos y 1 época en CPU, y así se verificó localmente que cada uno corre de punta a punta sin editar el notebook. Las métricas del modo smoke no tienen valor experimental.

## Decisiones de diseño — Ejercicio 1 (ESC-50: CNN 2D vs. Transformer A/B)

Notebook: `src/Adell_Giannice_Mazzola_Ej1_ESC50.ipynb` (generado por `src/_build/build_ej1.py`). Todos los hiperparámetros viven en el dict `HPARAMS` de la celda de configuración y se imprimen al correr el notebook.

### 1. Split: folds 1–3 train / fold 4 validación / fold 5 test

- **Elegido:** los folds oficiales de ESC-50; 1–3 (1200 clips) para entrenar, 4 (400) para validación y 5 (400) para test.
- **Alternativas:** validación cruzada de 5 folds (protocolo del paper de Piczak); split aleatorio estratificado.
- **Por qué:** lo pide la consigna. La 5-fold CV multiplica por cinco el costo (tres modelos × cinco folds en una sesión de Colab T4) y queda fuera de alcance (`non_goals`). Se usan los folds **oficiales** porque ESC-50 extrae varios clips de una misma grabación original (de Freesound) y los folds los agrupan: un split aleatorio pondría fragmentos de la misma grabación en train y test, lo que infla las métricas (fuga de información). El fold 4 separado permite hacer early stopping y elegir el mejor estado sin mirar el test; el fold 5 se evalúa una sola vez al final. El dataset es chico (2000 clips), así que se usa completo, como indica la consigna.

### 2. Carga del audio: `soundfile` + `torchaudio.transforms`

- **Elegido:** leer los WAV con `soundfile` y convertirlos a tensor; resampleo, STFT, mel, dB y SpecAugment con `torchaudio.transforms`.
- **Alternativas:** `torchaudio.load`; `librosa`.
- **Por qué:** en las versiones recientes de torchaudio `torchaudio.load` delega en `torchcodec`, que puede no estar instalado o no comportarse igual en Colab y en el entorno local. `soundfile` viene en Colab y lee WAV sin dependencias extra. Las transformaciones de torchaudio son `nn.Module` (como muestra la cátedra), así que corren en GPU y se integran al pipeline de PyTorch. La descarga del zip (~600 MB) y la extracción son idempotentes (si ya están, no se repiten; la descarga escribe a `.part` y renombra al terminar). En Colab los WAV se descomprimen en el disco local (`/content`), mucho más rápido que Drive; en Drive solo quedan el cache de mels y los checkpoints.

### 3. Frecuencia de muestreo 22050 Hz y duración fija de 5 s

- **Elegido:** resamplear de 44.1 kHz a 22050 Hz; recortar/completar con ceros a exactamente 5 s (110 250 muestras). Mono (promedio de canales).
- **Alternativas:** mantener 44.1 kHz; 16 kHz (típico de voz).
- **Por qué:** es la receta de ESC-50 de la cátedra (Unidad I). Por Nyquist, 22050 Hz representa hasta ~11 kHz, donde está la mayor parte de la energía relevante de los sonidos ambientales, y reduce a la mitad el costo de la STFT y el tamaño de los datos. 16 kHz recortaría a 8 kHz y perdería contenido de sonidos agudos (pájaros, vidrio, sirenas). La longitud fija permite un tensor de forma constante.

### 4. Parámetros de la STFT: `n_fft = 2048`, `hop = 512`, ventana Hann

- **Elegido:** ventana de 2048 muestras (~92.9 ms), hop de 512 (~23.2 ms, 75 % de solape), ventana Hann.
- **Resoluciones:** Δf = fs/n_fft = 22050/2048 ≈ 10.77 Hz; Δt = hop/fs ≈ 23.2 ms; 1 + 110250 // 512 = **216 frames** por clip (STFT centrada). El notebook calcula e imprime estos valores.
- **Alternativas:** `n_fft` 512/1024 (mejor resolución temporal, peor frecuencial); hop = n_fft/2 (50 % de solape, menos frames).
- **Por qué:** son los defaults de librosa/torchaudio para música/audio general a 22050 Hz y dan el espectrograma 128 × ~216 que menciona la consigna. La ventana de ~93 ms resuelve bien bandas cercanas en graves (motores, lluvia, zumbidos), y el hop de N/4 (75 % de solape, lo típico según la cátedra) mantiene una resolución temporal de ~23 ms, suficiente para los transitorios (golpes, tic-tac). La ventana Hann es el default razonable de la cátedra: reduce el *leakage* respecto de la rectangular. El trade-off tiempo-frecuencia (Gabor) se muestra en el notebook con dos ventanas (256 y 4096 muestras) sobre un sonido de golpes.

### 5. Mel-spectrogram con `n_mels = 128` en dB (`top_db = 80`)

- **Elegido:** `MelSpectrogram(power=2)` con 128 filtros triangulares y conversión a dB con `AmplitudeToDB(stype='power', top_db=80)`, aplicada por clip.
- **Alternativas:** STFT lineal (1025 bins); 40/64/80 bandas mel; MFCC.
- **Por qué:** receta de la cátedra (n_mels = 128, dB). La escala mel concentra resolución en graves y comprime agudos, igual que el oído, y reduce el eje de frecuencia ~8× respecto de la STFT lineal (1025 → 128). La escala dB alinea la intensidad con la percepción y comprime el rango dinámico, lo que estabiliza el entrenamiento. `top_db = 80` recorta el piso de ruido a 80 dB por debajo del máximo de cada clip (evita valores de −100 dB en silencios digitales). Los MFCC decorrelacionan pero descartan información que la CNN/Transformer puede aprovechar; la cátedra indica que hoy el mel crudo suele ser mejor con DL.

### 6. Precómputo y cache de los mels

- **Elegido:** calcular una sola vez el tensor `(2000, 128, 216)` en float32 (~220 MB), guardarlo en disco (`mels_full.pt`) y tenerlo en memoria (en GPU) durante el entrenamiento; batching manual por índices.
- **Alternativas:** `Dataset`/`DataLoader` que calcula el mel en cada época.
- **Por qué:** la representación es determinista, así que recalcularla en cada época es costo puro (la cátedra recomienda cachear transformaciones costosas). Con el tensor en GPU cada época tarda segundos. El cache está en Drive y se reutiliza si se reinicia la sesión; el smoke test usa un cache aparte.

### 7. Normalización con media y desvío de train

- **Elegido:** estandarización global (un escalar de media y uno de desvío) calculada **solo** sobre los mels de train y aplicada a los tres splits.
- **Alternativas:** normalización por clip; por banda mel; sin normalizar.
- **Por qué:** centra los valores en 0 con varianza ~1, lo que conviene a la proyección lineal de parches y a la LayerNorm/BatchNorm. Usar estadísticas de train evita filtrar información de validación/test. La normalización por clip borraría la diferencia de nivel entre sonidos fuertes y débiles, que es información útil. Como después de normalizar la media es 0, el relleno de SpecAugment y el *padding* de parches con ceros equivalen a "valor medio".

### 8. SpecAugment solo en la CNN y en la Config B

- **Elegido:** `FrequencyMasking(freq_mask_param=30)` + `TimeMasking(time_mask_param=40)` con `iid_masks=True` (máscara distinta por ejemplo), aplicado *on-the-fly* sobre cada batch de train, después del mel y antes del modelo; nunca en validación ni test.
- **Alternativas:** mixup, time shift, ruido gaussiano, time stretching / pitch shifting sobre la onda.
- **Por qué:** es la augmentation de la cátedra para mel-spectrograms (mismos parámetros 30/40) y de la receta de ESC-50. Es barata (opera sobre el tensor ya cacheado, en GPU) y ataca el principal problema del dataset: pocos ejemplos por clase (24 en train). La consigna pide SpecAugment para la CNN y para B, y **ninguna** regularización para A; así A vs. B aísla el efecto de la regularización.

### 9. Tokenización de la Config A: franjas temporales de banda completa → 12 tokens

- **Elegido:** parches de 128 × 18 (todas las bandas mel × 18 frames ≈ 418 ms); 216 / 18 = **12 tokens** (+ [CLS]).
- **Alternativas:** franjas de 128 × 16 (14 tokens) o 128 × 27 (8 tokens).
- **Por qué:** la consigna pide secuencias cortas de ~8–16 tokens formadas por franjas temporales de banda completa; 18 divide exactamente 216, así que no hay relleno. Cada token ve todo el espectro de un tramo de tiempo; el encoder solo modela relaciones temporales entre tramos.

### 10. Tokenización de la Config B: parches 2D 16 × 16 → 8 × 14 = 112 tokens

- **Elegido:** parches cuadrados de 16 × 16 sin solape, estilo AST/ViT; la grilla es 128/16 = 8 filas (frecuencia) × ⌈216/16⌉ = 14 columnas (tiempo) = **112 tokens** (+ [CLS]). Los 216 frames se completan con ceros hasta 224 a la derecha.
- **Alternativas:** recortar a 208 frames (8 × 13 = 104 tokens, la sugerencia del plan); parches con solape como AST original (stride 10 → >1000 tokens); parches 16 × 8.
- **Por qué:** la consigna pide ~50–200 tokens; 112 está en el medio del rango. Se eligió **rellenar** en vez de recortar para no descartar los últimos ~186 ms del clip (A usa todos los frames, así que recortar solo en B habría sido una diferencia extra entre configuraciones); el relleno con 0 equivale a la media tras la normalización. El solape de AST multiplicaría los tokens y el costo cuadrático de la atención sin preentrenamiento que lo justifique. Con parches 2D cada token es una región tiempo-frecuencia localizada, lo que además permite visualizar la atención sobre el espectrograma. *(Desvío respecto del plan: 112 tokens en vez de 104 por el relleno.)*

### 11. Arquitectura común del Transformer (`SpectrogramTransformer`)

- **Elegido:** una única clase parametrizada por `patch`, `d_model`, `n_heads`, `n_layers`, `d_ff`, `dropout` y `pre_norm`. Bloques escritos a mano a partir de las prácticas de la Unidad II: `scaled_dot_product_attention`, `MultiHeadAttention` (Wq/Wk/Wv/Wo), `PositionwiseFFN`, `EncoderLayer`. Proyección lineal de parches (`nn.Linear(ph·pw, d_model)`), token **[CLS]** aprendido, **positional embedding 1D aprendido**, cabeza lineal sobre el [CLS]. Una celda verifica con `isinstance` que ningún submódulo sea una clase prohibida de `torch.nn`.
- **Alternativas:** PE sinusoidal (el de la Practica1/4); *mean pooling* de los tokens (Practica4) en lugar de [CLS]; FFN con ReLU (cátedra).
- **Por qué:** una sola clase garantiza que A y B son "el mismo encoder". El [CLS] y el PE aprendido siguen a ViT/AST (Unidad III), y el [CLS] da un vector natural para la visualización de atención ([CLS] → parches). El PE aprendido se usa en **ambas** configuraciones para no introducir otra diferencia. GELU en la FFN sigue a ViT/AST; es un cambio menor respecto de la ReLU de la práctica. La atención devuelve sus pesos, como en la cátedra, para poder visualizarlos.

### 12. Post-norm (A) vs. pre-norm + warmup (B)

- **Elegido:** A usa post-norm (`x = LN(x + subcapa(x))`, Transformer original, como la `EncoderLayer` de la cátedra) con LR constante; B usa pre-norm (`x = x + subcapa(LN(x))`, ViT) con una LayerNorm final y LR con **warmup lineal de 5 épocas + decaimiento coseno** hasta 50 épocas (por paso).
- **Alternativas:** pre-norm sin warmup; warmup también para A.
- **Por qué:** lo fija la consigna. En pre-norm el camino residual no pasa por la normalización, lo que da gradientes mejor condicionados en capas profundas; el warmup evita pasos grandes al inicio, cuando la atención es casi uniforme y los gradientes son ruidosos. El coseno baja el LR hacia el final para afinar. A se deja deliberadamente "desnudo" (sin warmup ni ninguna otra ayuda) para que la comparación muestre el aporte del paquete de B.

### 13. Tamaño: d_model = 64, 2 capas, d_ff = 128; cabezas 2 (A) y 4 (B)

- **Elegido:** ambos encoders con 2 capas, `d_model = 64`, `d_ff = 128` (2 × d_model); A con 2 cabezas (d_k = 32) y B con 4 (d_k = 16).
- **Conteo de parámetros** (lo imprime el notebook, desglosado):
  - Encoder (2 capas): 66 944 en **ambas** (MHA 16 640 + FFN 16 576 + 2 LN 256 = 33 472 por capa). El número de cabezas no cambia los parámetros.
  - Config A: patch embed 147 520 (2304 → 64) + [CLS]/PE 896 + cabeza 3250 = **218 610** en total.
  - Config B: patch embed 16 448 (256 → 64) + [CLS]/PE 7296 + cabeza y LN final 3378 = **94 066** en total.
- **Alternativas:** B con 3 capas o d_model 128 (permitido por la consigna) para igualar el total; d_ff = 4 × d_model.
- **Por qué:** la consigna pide "mismo encoder (mismo tamaño)" y para A fija 2 capas, d = 64 y 2 cabezas; B usa los mismos valores dentro de sus rangos, con 4 cabezas (más subespacios de atención con el mismo costo). Así el **encoder** es idéntico en parámetros y la diferencia total se debe únicamente a la tokenización: un token de A tiene 2304 valores y uno de B 256, por lo que la proyección de A es 9× más grande. Igualar el total agrandando B mezclaría el efecto de la capacidad con el de la tokenización. `d_ff = 2·d_model` mantiene el modelo chico (el dataset tiene 1200 clips de train y un Transformer más grande solo sobreajusta más).

### 14. Dropout y weight decay

- **Elegido:** A: dropout 0 y weight decay 0 (AdamW con `weight_decay=0` equivale a Adam). B: dropout 0.1 (sobre embeddings, salidas de atención/FFN y capa oculta de la FFN) y weight decay 0.05.
- **Alternativas:** dropout 0.2; dropout también sobre los pesos de atención.
- **Por qué:** A sin ninguna regularización por consigna (incluye el weight decay, que también regulariza). Para B la consigna pide 0.1–0.2; se tomó 0.1 porque el modelo ya es chico y también recibe SpecAugment (sumar regularizaciones fuertes a un modelo de ~94k parámetros puede llevar a *underfitting*). 0.05 es el weight decay típico de ViT/DeiT con AdamW.

### 15. CNN 2D

- **Elegido:** 4 bloques Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2 con 32/64/128/256 canales, Global Average Pooling, dropout 0.3 y capa lineal a 50 clases (**401 650** parámetros). AdamW, LR 1e-3 constante, weight decay 1e-2, SpecAugment.
- **Alternativas:** una CNN del tamaño de los Transformers (16–128 canales, ~100k parámetros); dos convoluciones por bloque (estilo VGG); ResNet.
- **Por qué:** es el bloque que presenta la cátedra (Unidad I) y el baseline "CNN 2D simple" (~70 % en ESC-50) contra el que se compara. Cuatro poolings llevan 128 × 216 a 8 × 13 con un campo receptivo que cubre bandas y tramos amplios; el GAP da invariancia a la posición del evento y tiene muy pocos parámetros en la cabeza. Se prefirió un baseline fuerte (más canales) antes que igualar parámetros: el objetivo es contrastar los Transformers contra una CNN razonable; la eficiencia (accuracy por parámetro y tiempo) se reporta en las conclusiones. BatchNorm y dropout son la regularización natural de la CNN, junto con SpecAugment.

### 16. Optimizador, learning rate, batch, épocas y early stopping

- **Elegido:** AdamW para los tres; batch 32; LR 1e-3 (CNN) y 5e-4 (A y B); máximo **50 épocas**; early stopping con **paciencia 10** sobre la **accuracy de validación**; se guarda el mejor `state_dict` y es el que se evalúa en test. En `SMOKE_TEST`: 1 época, paciencia 1 y warmup de 1 época.
- **Alternativas:** Adam/SGD con momentum; early stopping sobre la pérdida de validación; paciencia 5.
- **Por qué:** AdamW es el estándar para Transformers (desacopla el weight decay) y funciona bien para CNNs chicas. Batch 32 da ~38 pasos por época con 1200 clips: suficientes actualizaciones sin ruido excesivo. Los Transformers usan un LR menor que la CNN porque son más sensibles a pasos grandes (sin BatchNorm); A y B comparten el LR pico para que solo cambie el *schedule*. La consigna habla de 20–50 épocas con early stopping; 50 como tope y paciencia 10 dejan que la curva de validación se estabilice sin cortar por fluctuaciones de un fold de 400 clips. Se monitorea la accuracy (la métrica reportada) en vez de la pérdida porque en clasificación con pocos datos la pérdida de validación puede subir por sobreconfianza mientras la accuracy sigue mejorando.

### 17. Medición del tiempo y checkpoints

- **Elegido:** tiempo de pared con `time.time()` alrededor del loop de entrenamiento (incluye la validación por época, excluye el precómputo), con `torch.cuda.synchronize()` al inicio y al final. Checkpoint por modelo en Drive (mejor estado + historial + tiempo); si existe, se carga en vez de reentrenar (en `SMOKE_TEST` se reentrena siempre, en una subcarpeta aparte).
- **Por qué:** el estilo de la cátedra mide con `time.time()`; sincronizar evita subestimar el tiempo por la ejecución asíncrona de CUDA. El checkpoint permite retomar si Colab corta la sesión sin perder los modelos ya entrenados. Como los tres modelos usan el mismo loop, los mismos datos en memoria y la misma GPU, los tiempos son comparables entre sí (no con otras máquinas).

### 18. Métricas y matriz de confusión

- **Elegido:** accuracy y F1-macro (`sklearn`) en el fold 5; matriz de confusión 50 × 50 normalizada por fila como heatmap para cada modelo; tabla pandas con accuracy, F1, mejor accuracy de validación, accuracy de train en la mejor época, parámetros, tiempo y épocas; lista de los pares más confundidos del mejor modelo.
- **Por qué:** lo pide la consigna. Con clases balanceadas accuracy y F1-macro son parecidos, pero el F1-macro penaliza a un modelo que ignora clases enteras. Normalizar por fila hace la matriz legible como "fracción de cada clase real". La brecha train–validación en la mejor época cuantifica el sobreajuste de A frente a B.

### 19. Visualización de la atención de la Config B

- **Elegido:** (1) atención del token [CLS] hacia los parches en la **última capa**, promediada sobre las 4 cabezas; (2) **attention rollout** (Abnar & Zuidema, 2020): por capa, promedio de cabezas + 0.5·identidad (conexión residual), renormalización por filas y producto de las matrices de todas las capas. El vector de 112 pesos se lleva a la grilla 8 × 14, se sobremuestrea con *nearest* (cada peso cubre su parche 16 × 16), se recorta a 128 × 216 y se superpone al mel en escala de grises. Se muestran dos aciertos y dos errores de test, y el mapa de atención medio sobre todo el test (con el reparto por bandas bajas/medias/altas).
- **Alternativas:** una sola cabeza; atención media de todos los tokens (Practica4); Grad-CAM; interpolación bilineal.
- **Por qué:** con un token [CLS], su fila de atención es la forma directa de ver "a qué parches mira" la representación que se clasifica (Unidad III: el [class] termina atendiendo a las regiones relevantes). Una sola capa ignora cómo se mezcla la información entre capas; el rollout lo aproxima. *Nearest* respeta que el peso pertenece a todo el parche (la bilineal inventaría estructura dentro del parche). Mostrar aciertos y errores permite ver si los errores se asocian a atender a regiones de fondo/silencio.

### 20. Semilla y reproducibilidad

- **Elegido:** `SEED = 42` en `random`, `numpy` y `torch` (y `cuda.manual_seed_all`); `set_seed()` se vuelve a llamar antes de construir cada modelo y antes de cada entrenamiento, así la inicialización y el orden de los batches no dependen de qué se corrió antes. Se imprimen las versiones de las librerías y el dict `HPARAMS`.
- **Por qué:** lo pide la consigna. Re-sembrar por modelo hace que cada entrenamiento sea reproducible por separado (por ejemplo, si se retoma desde checkpoints). No se fuerzan algoritmos deterministas de cuDNN: pueden quedar diferencias mínimas entre corridas en GPU, a cambio de velocidad.

### 21. Modo `SMOKE_TEST`

- **Elegido:** `TP1_SMOKE_TEST=1` → 5 clases (una por macro-categoría: dog, rain, crying_baby, door_wood_knock, helicopter) × 2 clips por fold (30/10/10 clips), 1 época; los modelos mantienen las 50 salidas; caches y checkpoints en la subcarpeta `smoke/`.
- **Por qué:** permite verificar el notebook de punta a punta en CPU en pocos minutos sin tocar el código; las métricas del smoke no son interpretables (y el notebook lo avisa al imprimir las conclusiones).

## Decisiones de diseño — Ejercicio 2 (NER con DistilBERT sobre CoNLL-2003, LoRA vs. full)

Notebook: `src/Adell_Giannice_Mazzola_Ej2_NER_CoNLL.ipynb` (generado por `src/_build/build_ej2.py`).
Cada sección dice qué se eligió, qué alternativas había y por qué.

### 1. Stack y estructura

- **Elegido:** HF `transformers` (`AutoModelForTokenClassification`, `Trainer`, `TrainingArguments`), `datasets`, `peft` (`LoraConfig`, `get_peft_model`, `TaskType.TOKEN_CLS`), `evaluate` + `seqeval`. En Colab se instala con `%pip install -q -U ...` y se desinstala `torchao`, igual que en NB2.
- **Alternativas:** loop de entrenamiento propio en PyTorch; `SFTTrainer` de `trl`.
- **Por qué:** es el patrón de NB2 de la cátedra. `Trainer` ya trae evaluación y guardado por época, fp16, reanudación y selección del mejor checkpoint, así que los tres modos comparten exactamente el mismo bucle. `SFTTrainer` es para LLMs generativos y no aplica a clasificación de tokens. La versión de Colab de `torchao` rompe la importación de `peft`.
- **Compatibilidad con transformers 5.x** (lo que instala `pip -U`): `Trainer(processing_class=tokenizer)` en lugar de `tokenizer=`, `eval_strategy` en lugar de `evaluation_strategy`, y `warmup_steps=0.1` (un float < 1 se interpreta como proporción de los pasos totales), porque `warmup_ratio` ya no existe en `TrainingArguments`.

### 2. Carga del dataset vía parquet

- **Elegido:** `load_dataset('eriktks/conll2003', revision='refs/convert/parquet')`.
- **Alternativas:** `load_dataset('eriktks/conll2003')` a secas; `trust_remote_code=True`; bajar los archivos CoNLL originales y parsearlos a mano.
- **Por qué:** el repo solo tiene un script de carga (`conll2003.py`), y `datasets` ≥ 4 ya no ejecuta scripts (tampoco con `trust_remote_code`). La rama `refs/convert/parquet` es la conversión automática que publica el Hub del mismo dataset, con los mismos splits (train 14041 / validation 3250 / test 3453) y las features `ClassLabel`. Los nombres de las etiquetas se leen de `features['ner_tags'].feature.names` (verificado: `O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC`). Así no se hardcodea un orden que podría no coincidir.

### 3. Dataset completo y splits oficiales

- **Elegido:** los tres splits oficiales completos. Validation sirve para seleccionar el checkpoint y test solo para el reporte final.
- **Alternativas:** submuestrear train; hacer un split propio.
- **Por qué:** CoNLL-2003 es chico (unas 14k oraciones) y una época de DistilBERT sobre todo el train tarda pocos minutos en una T4, así que no hace falta recortar. Los splits oficiales hacen comparables los resultados con la literatura, y test no participa de ninguna decisión, ni siquiera del early stopping.
- **Smoke test:** 64/32/32 oraciones (shuffle con la semilla) y 1 época, solo para verificar que el código corre en CPU.

### 4. Tokenización, `max_length` y alineación de etiquetas

- **Elegido:** `tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=128)`. Solo el **primer subtoken** de cada palabra lleva la etiqueta; los subtokens siguientes y `[CLS]`/`[SEP]` llevan `-100`. El padding es dinámico con `DataCollatorForTokenClassification`.
- **Alternativas:** etiquetar todos los subtokens (propagar `B-X` como `I-X` a los siguientes); usar el último subtoken; padding fijo a 512.
- **Por qué:**
  - `is_split_into_words=True` respeta la segmentación en palabras del dataset, y `word_ids()` da el mapeo subtoken→palabra.
  - Etiquetar solo el primer subtoken deja exactamente una predicción por palabra. Eso hace que la pérdida y la métrica se computen sobre las mismas unidades que el gold estándar de CoNLL (palabras), sin sobreponderar palabras largas. Es la convención del paper de BERT (Devlin et al., 2019) y del ejemplo oficial de HF. `-100` es el `ignore_index` de `CrossEntropyLoss`.
  - `max_length=128`: el notebook imprime la distribución de largos en subtokens. Las oraciones de CoNLL son cortas y prácticamente ninguna supera 128 (el porcentaje se calcula por código), así que se evita el costo de secuencias largas sin perder etiquetas.
  - El padding dinámico por batch evita procesar tokens de relleno. El collator rellena también `labels` con `-100`.
- **Modelo cased:** en NER la mayúscula inicial es una señal fuerte de nombre propio, y además lo pide la consigna (`distilbert-base-cased`).

### 5. Métrica: seqeval a nivel entidad

- **Elegido:** `evaluate.load('seqeval')`, con precision, recall y **F1 micro a nivel entidad** (métrica principal y `metric_for_best_model='f1'`) más la accuracy por palabra. También se reporta `classification_report` por tipo de entidad.
- **Alternativas:** accuracy por token; F1 por token con sklearn.
- **Por qué:** es la métrica oficial del shared task de CoNLL-2003. Una entidad cuenta como acierto solo si coinciden su tipo y sus límites exactos. La accuracy por token está inflada por la etiqueta `O`, que ocupa la gran mayoría de las palabras (el notebook imprime el porcentaje), así que un modelo casi trivial ya da una accuracy alta. Se incluye igual porque la consigna pide "F1/accuracy". Se evalúa solo sobre posiciones con etiqueta distinta de `-100`, es decir, una por palabra. `zero_division=0` evita warnings y NaN cuando un modelo no predice alguna clase (pasa en el smoke test).

### 6. "Antes" del fine-tuning = linear probe de 1 época

- **Elegido:** el mismo modelo con todo `distilbert.*` congelado; solo se entrena `classifier` (768×9 + 9 = 6921 parámetros) durante 1 época, con LR 1e-3.
- **Alternativas:** evaluar el modelo con la cabeza recién inicializada (sin entrenar); no reportar un "antes".
- **Por qué:** con una cabeza aleatoria el resultado es azar sobre 9 clases y no dice nada del modelo preentrenado. El linear probe mide cuánta información de NER ya está disponible linealmente en las features preentrenadas (el concepto de linear probe de la Unidad III). Los deltas full−probe y LoRA−probe miden entonces la ganancia real de **adaptar el backbone**, que es lo que distingue a full y LoRA. Con 1 época alcanza porque es un problema convexo en la cabeza. El LR es alto (1e-3) porque solo se entrena una capa lineal nueva, como en LoRA.

### 7. LoRA: r, alpha, dropout, targets y `modules_to_save`

- **Elegido:** `LoraConfig(task_type=TaskType.TOKEN_CLS, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q_lin','v_lin'], modules_to_save=['classifier'])`.
- **Alternativas:** r = 4 o 16; alpha = r; aplicarlo también a `k_lin`, `out_lin` y a las FFN (`lin1`, `lin2`); no entrenar la cabeza.
- **Por qué:**
  - **Targets `q_lin`/`v_lin`:** son las proyecciones query y value de la atención de DistilBERT. Es donde el paper de LoRA (Hu et al., 2021) lo aplica, lo usa NB2 para DistilBERT y lo pide la consigna ("target modules de atención").
  - **r = 8:** está en el rango 4–16 que la cátedra indica como suficiente. Da 8·(768+768) = 12 288 parámetros por matriz, 6 capas × 2 matrices = 147 456, un 0,2% del modelo. El notebook verifica esta cuenta contra el conteo real.
  - **alpha = 16 (α/r = 2):** es la configuración de NB2. El factor α/r desacopla la magnitud del update del rango; la práctica habitual es fijarlo y ajustar el LR.
  - **dropout 0.05:** una regularización leve del camino LoRA, como en NB2.
  - **`modules_to_save=['classifier']`:** la cabeza de NER es nueva (inicializada al azar), así que tiene que entrenarse. Si no, LoRA tendría que aprender a mapear a una proyección aleatoria fija. `modules_to_save` la deja entrenable y la guarda junto al adaptador. Se cuenta dentro de los parámetros entrenables de LoRA, y el probe entrena exactamente la misma cabeza.

### 8. Learning rates

- **Elegido:** full 2e-5; LoRA 1e-3; probe 1e-3.
- **Alternativas:** full entre 3e-5 y 5e-5; LoRA entre 1e-4 y 5e-4.
- **Por qué:** son los valores de NB2 (2e-5 full / 1e-3 LoRA). En full se mueven pesos preentrenados, y un LR bajo evita el olvido catastrófico. En LoRA, B arranca en 0 y los adaptadores tienen pocos parámetros, así que necesitan un LR mayor para moverse en pocas épocas (la cátedra lo señala: "LR de LoRA mayor que en full"). El LR es la **única** diferencia de hiperparámetros entre full y LoRA que permite la consigna.

### 9. Batch, épocas, scheduler, warmup, weight decay

- **Batch 16** (train) / 32 (eval): está dentro del rango 16–32 pedido y es el valor de NB2. Con secuencias de ≤128 tokens entra holgadamente en una T4 incluso en full. El batch de eval solo afecta la velocidad.
- **3 épocas** para full y LoRA (lo pide la consigna). En CoNLL, 3 épocas bastan para que el F1 de validación se estabilice, y más épocas tienden a sobreajustar el full.
- **Warmup 10% + scheduler coseno:** es la receta de la Unidad IV (diapositiva "Entrenamiento": warmup, scheduler coseno y evaluación periódica). El warmup evita pasos grandes cuando los estados de Adam todavía no son confiables, algo importante sobre todo para el LR alto de LoRA.
- **Weight decay 0.01:** es el valor estándar de BERT y del ejemplo de HF.
- **Semilla 42** en `random`, `numpy`, `torch` y `transformers.set_seed`, y `set_seed` de nuevo antes de construir cada modelo, para que la cabeza se inicialice igual en los tres modos.

### 10. Early stopping y `load_best_model_at_end`

- **Elegido:** `eval_strategy='epoch'`, `save_strategy='epoch'`, `load_best_model_at_end=True`, `metric_for_best_model='f1'` y `EarlyStoppingCallback(early_stopping_patience=1)`. Test se evalúa con el mejor checkpoint de validación.
- **Por qué:** así se cumple "entrenar hasta que la validación se estabilice" dentro del presupuesto de 3 épocas. Si una época no mejora el F1 de validación, se corta, y en cualquier caso se usa el mejor punto, no el último. Se selecciona por F1 de entidad (la métrica de la tarea), no por la loss. El criterio es idéntico para los tres modos.

### 11. Precisión mixta

- **Elegido:** `fp16=torch.cuda.is_available()`.
- **Por qué:** la T4 no tiene soporte eficiente de bf16 y sí tiene Tensor Cores fp16. fp16 reduce la memoria de activaciones y acelera el entrenamiento (lo pide la consigna). En CPU (smoke test) fp16 no está soportado por `Trainer`, así que se desactiva.

### 12. Checkpoints, reanudación y caché de resultados

- **Elegido:** cada modo escribe en `BASE/out_<modo>` (Drive: `MyDrive/Colab_Notebooks/TP1_Adell_Giannice_Mazzola/ej2/`) con `save_total_limit=2`. Si ya hay checkpoints, `trainer.train(resume_from_checkpoint=get_last_checkpoint(...))` reanuda. Al terminar se guardan el modelo final (`ckpt_<modo>`) y `resultados_<modo>.json` (métricas, predicciones de test, `log_history`, tiempos). Si el JSON existe, el modo no se reentrena (`REENTRENAR=True` lo fuerza).
- **Por qué:** en Colab Free la sesión se puede cortar. Guardar por época permite retomar sin perder el trabajo, y el JSON evita repetir 3 corridas completas al reabrir el notebook. `save_total_limit=2` acota el uso de Drive (cada checkpoint full con estados de Adam pesa unos 0,8 GB); `Trainer` conserva siempre el mejor checkpoint aunque exceda el límite. En el smoke test se borra `out_<modo>` para entrenar siempre desde cero.
- **Limitación conocida:** si se reanuda, el tiempo y la memoria medidos corresponden solo a la sesión en la que terminó el entrenamiento.

### 13. Medición de memoria

- **Elegido:** `torch.cuda.empty_cache()` + `torch.cuda.reset_peak_memory_stats()` justo antes de `trainer.train()`, y `torch.cuda.max_memory_allocated()` al terminar. Se reporta "N/A" sin CUDA.
- **Por qué:** mide el pico de memoria asignada por tensores durante el entrenamiento: pesos, gradientes, estados de Adam y activaciones. Ahí está la diferencia teórica entre full y LoRA, porque LoRA no guarda gradientes ni momentos de Adam para los pesos congelados. `max_memory_allocated` es determinista y no incluye la caché del allocator, a diferencia de `nvidia-smi` o `max_memory_reserved`. Se resetea antes de cada modo para que las mediciones no se contaminen entre sí, y se liberan el modelo y el trainer al terminar cada experimento.

### 14. Medición del tamaño del checkpoint

- **Elegido:** full y probe con `trainer.save_model(ckpt_dir)` (modelo completo en safetensors + config + tokenizer); LoRA con `model.save_pretrained(ckpt_dir)`, que guarda solo el adaptador y la cabeza (`modules_to_save`). El tamaño es la suma de los archivos de la carpeta (`dir_size_mb` con `os.walk`, como en NB2) y se muestra en GB para full y en MB para LoRA, como pide la consigna.
- **Por qué:** es lo que realmente hay que almacenar o distribuir por tarea. El adaptador referencia el backbone por su id del Hub, y en la sección de inferencia se demuestra que se carga con `PeftModel.from_pretrained(modelo_base, adaptador)`. Se mide el modelo final, no los checkpoints intermedios, que además incluyen los estados del optimizador.

### 15. Parámetros entrenables

- **Elegido:** `print_trainable_parameters()` para LoRA (lo pide la consigna) y el conteo equivalente `sum(p.numel() for p in m.parameters() if p.requires_grad)` para los tres modos, con total y porcentaje.
- **Por qué:** así los tres modos se comparan con la misma función. El notebook además verifica la fórmula r·(d+k) de LoRA contra el conteo real.

### 16. Qué significa "mismas condiciones"

Full y LoRA (y el probe) comparten: dataset y splits, tokenizador y `max_length`, collator, batch de train y de eval, cantidad de épocas (salvo el probe, que por definición es de 1 época), scheduler y warmup, weight decay, optimizador (AdamW por defecto de `Trainer`), semilla, fp16, estrategia de evaluación y guardado, criterio de selección del mejor modelo y early stopping, y el mismo split de test para el reporte. Todos los `TrainingArguments` salen de una única función `make_args(mode, lr, epochs)`. Solo cambian el **learning rate** y **qué parámetros se entrenan**, que es justamente lo que se quiere comparar.

### 17. Reporte y conclusiones

- La tabla comparativa (pandas) incluye F1, accuracy, precision y recall en test, deltas contra el probe, parámetros entrenables (absolutos y %), memoria pico, tamaño del checkpoint (texto en GB/MB y columna numérica en MB), épocas, LR y tiempo. Los gráficos de barras usan escala log para parámetros y tamaño, porque las diferencias son de órdenes de magnitud.
- El reporte por entidad (seqeval) muestra si la brecha full−LoRA es uniforme o se concentra en algún tipo.
- La inferencia sobre una oración de ejemplo carga los modelos desde disco para demostrar el flujo real de uso del adaptador.
- Las conclusiones son f-strings calculadas sobre las métricas reales (ranking, brecha de F1, porcentaje de la mejora recuperada por LoRA, ratio de parámetros, ratio de tamaño, ahorro de memoria, tiempo, brecha por entidad). El markdown no contiene ningún número de resultado.

## Decisiones de diseño — Ejercicio 3 (Swin-Tiny en EuroSAT: LoRA vs. full fine-tuning)

Notebook: `src/Adell_Giannice_Mazzola_Ej3_Swin_EuroSAT.ipynb` (generado por `src/_build/build_ej3.py`).
Cada sección dice qué se eligió, qué alternativas había y por qué se eligió así.

### 1. Dataset: `timm/eurosat-rgb` y sus splits

- **Qué:** `load_dataset("timm/eurosat-rgb")`, versión RGB de EuroSAT (Helber et al., 2019): 10 clases, imágenes de 64×64, splits oficiales train 16.200 / validation 5.400 / test 5.400.
- **Alternativas:** el EuroSAT original (zip de 27.000 imágenes sin splits), la versión multiespectral de 13 bandas, u otras copias del Hub con un script de carga.
- **Por qué:** está en parquet (carga directa con `datasets` ≥ 4, sin scripts remotos), tiene `label` como `ClassLabel` con los nombres de las clases y trae splits fijos, así validación y test son reproducibles y comparables con otros trabajos. RGB coincide con los 3 canales con que se preentrenó Swin en ImageNet; las 13 bandas obligarían a cambiar el embedding de parches y a perder parte del preentrenamiento.

### 2. Subset estratificado de 10k para train, igual para todos los métodos

- **Qué:** `ds['train'].train_test_split(train_size=10_000, stratify_by_column='label', seed=42)['train']`, calculado **una sola vez** en el objeto `train_sub` y usado por probe, full y LoRA (los tres `Trainer` reciben el mismo `train_t = train_sub.with_transform(...)`). Validación y test son los splits oficiales completos.
- **Alternativas:** usar los 16.200 de train; un subset aleatorio sin estratificar; un subset distinto por método.
- **Por qué:** lo pide la consigna y además acota el tiempo de 3 épocas × 2 métodos en una T4. Estratificar conserva las proporciones de clase del train completo (el notebook imprime la diferencia máxima de proporción); con semilla fija el subset es reproducible (se imprimen los primeros `image_id`). Usar el mismo objeto es la condición central de "mismas condiciones": cualquier diferencia entre full y LoRA no puede venir de los datos.
- **Smoke test:** 100 / 50 / 50 imágenes (estratificadas con la misma función) y 1 época, solo para verificar el código en CPU.

### 3. Resolución: upscale 64 → 224 con los parámetros del image processor

- **Qué:** se carga `AutoImageProcessor` del checkpoint y de él se leen el tamaño (224), la media y el desvío (ImageNet) y la interpolación (bicúbica). Con esos valores se arman los pipelines de torchvision. Se verifica numéricamente que el pipeline de evaluación coincide con la salida del processor.
- **Alternativas:** usar el processor directamente en el `with_transform` (sin augmentation); entrenar a 64×64.
- **Por qué:** Swin-Tiny tiene ventana 7 y 4 etapas con patch merging; a 224 el mapa de la última etapa es 7×7 = una ventana, que es la configuración del preentrenamiento. A 64×64 el mapa final quedaría de 2×2, más chico que la ventana, y la tabla de sesgo posicional relativo y las features aprendidas quedarían fuera de su régimen. Leer media/desvío/tamaño del processor evita valores hardcodeados y garantiza que la entrada tenga la misma distribución que en el preentrenamiento. Se usa torchvision (y no solo el processor) porque la consigna pide augmentation on-the-fly, que el processor no hace.

### 4. Augmentation

- **Qué (train):** `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(15)`, `ToTensor`, `Normalize`. **Eval:** `Resize(224)` + `Normalize`. Se aplica con `with_transform`.
- **Alternativas:** sin augmentation; RandAugment/AutoAugment; mixup/cutmix; rotaciones de 90°; jitter de color.
- **Por qué:** las imágenes satelitales se toman desde arriba: no tienen orientación canónica, así que los flips verticales y las rotaciones producen vistas tan válidas como la original (a diferencia de fotos naturales, donde un flip vertical sería antinatural). El recorte con escala 0,8–1 simula pequeños cambios de encuadre sin perder la clase (una parcela de 64 px necesita la mayor parte del parche para reconocerse). Rotación leve (±15°) y no fuerte para limitar las esquinas negras que introduce el relleno. No se usa jitter de color porque el color es una señal fuerte en EuroSAT (agua vs. vegetación vs. cultivo) y alterarlo puede confundir clases. Es "augmentation básica", como pide la consigna. `with_transform` aplica las transformaciones al vuelo, sin materializar 10k tensores de 3×224×224 (~6 GB en float32) en la RAM estándar de Colab.

### 5. Modelo y cabeza

- **Qué:** `AutoModelForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224', num_labels=10, id2label, label2id, ignore_mismatched_sizes=True)`.
- **Por qué:** lo fija la consigna. `ignore_mismatched_sizes=True` descarta la cabeza de 1000 clases de ImageNet y crea una lineal nueva de 768 → 10 (transferencia reemplazando la cabeza, como en la Unidad III). `id2label/label2id` dejan los nombres de clase en la config guardada.

### 6. "Antes" del fine-tuning: linear probe de 1 época

- **Qué:** se congelan todos los parámetros `swin.*` y se entrena solo `classifier` (7.690 parámetros) durante 1 época con LR 1e-3; su accuracy en test es la métrica "antes" y se reportan los deltas full − probe y LoRA − probe.
- **Alternativas:** evaluar el modelo con la cabeza recién inicializada (accuracy ≈ azar, 10%); kNN sobre features.
- **Por qué:** una cabeza aleatoria no mide nada del backbone y hace que cualquier método parezca una mejora enorme. El linear probe (Unidad III) mide la calidad intrínseca de las features preentrenadas en esta tarea, así el delta refleja cuánto aporta *adaptar el backbone*. 1 época alcanza para un clasificador lineal sobre features fijas y lo mantiene barato; el LR alto (1e-3) es adecuado porque solo se ajusta una capa lineal desde cero.

### 7. LoRA: dónde, rango y cabeza

- **Qué:** `LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=<query, value>, modules_to_save=['classifier'])` + `get_peft_model` + `print_trainable_parameters()`.
- **Nombres de módulos (desvío documentado):** la consigna pide `target_modules=['query','value']`, que son los nombres de Swin en `transformers` 4.x (`attention.self.query/value`). En `transformers` 5.x (la que instala `pip -U` en Colab; verificado con 5.17) los módulos se renombraron a `attention.q_proj/k_proj/v_proj/o_proj`, y `['query','value']` hace fallar a `peft` con `NoMatchingPeftModuleError`. El notebook inspecciona `named_modules()`, imprime las capas lineales de la atención y elige `['query','value']` si existen o `['q_proj','v_proj']` si no. En ambos casos se adaptan exactamente las proyecciones **query y value** de todos los bloques Swin, que es lo que pide la consigna.
- **Por qué query/value:** es la elección original de Hu et al. (2021) y la de la cátedra (`q_lin`/`v_lin` en DistilBERT); la consigna pide LoRA sobre la atención.
- **Por qué r=8, α=16:** la cátedra indica r en 4–16 y usa r=8/α=16 en NB2; α/r = 2 escala la actualización sin tener que retocar el LR. Es el mismo par que en el Ej. 2, así las conclusiones sobre eficiencia son comparables entre ejercicios.
- **Por qué dropout 0,05:** regularización leve del camino LoRA, valor de la cátedra.
- **Por qué `modules_to_save=['classifier']`:** la cabeza es nueva (inicializada al azar) y tiene que entrenarse completa; sin esto quedaría congelada y LoRA solo podría adaptar features hacia una cabeza aleatoria. Además hace que `save_pretrained` guarde la cabeza junto al adaptador, así el checkpoint de LoRA es autosuficiente sobre el modelo base.

### 8. Learning rates

- **Qué:** full 5e-5, LoRA 1e-3, probe 1e-3.
- **Alternativas:** 2e-5 para full (valor de NB2 para DistilBERT); 5e-4 o 5e-3 para LoRA.
- **Por qué:** full ajusta todos los pesos preentrenados; un LR bajo evita destruir las features (olvido catastrófico). 5e-5 es el valor estándar para fine-tuning de ViT/Swin en clasificación de imágenes y, con solo 3 épocas y ~940 pasos, 2e-5 converge más lento. LoRA entrena pocos parámetros que parten de cero (B=0), por eso usa un LR mayor, como la cátedra (1e-3). Es la única diferencia de optimización entre full y LoRA, como exige la consigna.

### 9. Batch, épocas, scheduler, precisión y demás `TrainingArguments`

- **Batch 32** (eval 64): dentro del rango 16–32 de la consigna; Swin-Tiny a 224 con fp16 entra holgado en los 16 GB de la T4 incluso en full fine-tuning, y el batch más grande reduce el tiempo por época.
- **3 épocas** para full y LoRA, evaluación y guardado por época, `load_best_model_at_end=True` con `metric_for_best_model='accuracy'`: lo pide la consigna; equivale a early stopping por selección del mejor checkpoint de validación.
- **Warmup lineal (10% de los pasos) + decaimiento coseno**, weight decay 0,01 (AdamW por defecto del `Trainer`): receta de la Unidad IV; el warmup estabiliza los primeros pasos (sobre todo con la cabeza aleatoria y el LR alto de LoRA). En `transformers` 5 `warmup_ratio` ya no existe: se pasa `warmup_steps=0.1`, que ahora acepta una fracción de los pasos totales.
- **fp16 solo si hay CUDA** (`USE_FP16 = torch.cuda.is_available()`): la T4 tiene tensor cores fp16 y ahorra memoria; en CPU fp16 no está soportado por el `Trainer`.
- **`remove_unused_columns=False`**: el `Trainer` borraría la columna `image` antes de que llegue al `with_transform`. **`label_names=['labels']`**: el `PeftModel` tiene un `forward(*args, **kwargs)` genérico y el `Trainer` no puede inferir el nombre de las etiquetas; fijarlo en los tres experimentos mantiene los argumentos idénticos.
- **Collator** propio que apila `pixel_values` y arma `labels`, porque las imágenes ya vienen como tensores del transform.
- **`processing_class=processor`** (API de `transformers` ≥ 5, reemplaza a `tokenizer=`): guarda la config del processor junto a cada checkpoint.
- **`dataloader_pin_memory` solo con CUDA**: sin GPU no tiene efecto y solo genera avisos.
- **`dataloader_num_workers=2` en Colab, 0 fuera**: en Colab paraleliza el decodificado y el resize; en Windows (smoke local) los workers requieren pickling de funciones definidas en el notebook.
- **`save_total_limit=2`**: con `load_best_model_at_end` conserva el mejor y el último checkpoint, suficiente para reanudar sin llenar Drive.
- **`make_args` único**: probe, full y LoRA construyen sus argumentos con la misma función; solo cambian `learning_rate` y `num_train_epochs` (1 en el probe).

### 10. Reanudación y checkpoints en Drive

- **Qué:** checkpoints por época en `MyDrive/Colab_Notebooks/TP1_Adell_Giannice_Mazzola/ej3/out_{mode}`; si existe uno, `trainer.train(resume_from_checkpoint=get_last_checkpoint(out_dir))`. Al terminar cada experimento se guarda `resultados_{mode}.json` con métricas, `log_history`, predicciones de test, tiempo y memoria; si ya existe, el experimento no se reentrena.
- **Por qué:** una sesión de Colab puede cortarse en medio de un entrenamiento de varios minutos; la reanudación evita perder épocas y el JSON evita repetir experimentos terminados (las secciones de comparación trabajan solo con los JSON). El notebook avisa si un experimento se reanudó, porque en ese caso tiempo y memoria corresponden solo al último tramo. El smoke test usa la subcarpeta `smoke/` y la borra al empezar, para correr siempre completo.

#### Recarga correcta de checkpoints (workaround de `transformers` 5.x)

- **Problema detectado en el smoke test (transformers 5.17):** `save_pretrained` guarda Swin con los nombres de pesos legacy (`attention.self.query`, `intermediate.dense`, ...), pero `Trainer._load_best_model` y `Trainer._load_from_checkpoint` recargan el `.safetensors` con `load_state_dict(strict=False)` **sin** convertir los nombres a los del modelo en memoria (`attention.q_proj`, `mlp.fc1`, ...). El backbone no se carga y solo se emite un aviso. En full fine-tuning, `load_best_model_at_end` dejaría el backbone de la última época con la cabeza de la mejor época, y al reanudar desde un checkpoint el entrenamiento seguiría con el backbone preentrenado.
- **Solución:** para probe y full, `cargar_pesos(model, ckpt)` recarga el checkpoint con `AutoModelForImageClassification.from_pretrained` (que sí aplica la conversión) y copia el `state_dict` con `strict=True`; se aplica antes de `trainer.train(resume_from_checkpoint=...)` y después de entrenar sobre `trainer.state.best_model_checkpoint`. Se mantiene `load_best_model_at_end=True` porque es el que registra y conserva el mejor checkpoint. Se verificó por separado que, después de `cargar_pesos`, la carga parcial del `Trainer` deja los pesos idénticos a los del checkpoint (sin el fix no lo están). LoRA no se ve afectado: los adaptadores se guardan y recargan con `peft`. El aviso del `Trainer` se silencia (`transformers.trainer` en nivel ERROR) porque ya está resuelto.

### 11. Métricas

- **Qué:** accuracy (métrica principal, de la consigna) y F1-macro con `evaluate`; `classification_report` por clase y matriz de confusión del mejor modelo; ejemplos mal clasificados y pares de clases más confundidos.
- **Por qué:** EuroSAT está casi balanceado, así que la accuracy es representativa; el F1-macro confirma que no se esconde una clase mala detrás del promedio. La matriz de confusión muestra los errores esperables entre clases parecidas (p. ej. cultivos/pasturas, autopista/río).
- **Selección del mejor modelo** para la matriz: el de mayor accuracy en test entre full y LoRA, elegido por código.

### 12. Parámetros entrenables, memoria, tamaño y tiempo

- **Parámetros:** `print_trainable_parameters()` para LoRA y conteo equivalente (`n_trainable`/`n_total`) para probe y full; se reportan en absoluto y en %.
- **Memoria:** `torch.cuda.reset_peak_memory_stats()` antes de `trainer.train()` y `torch.cuda.max_memory_allocated()` después; antes de cada experimento se libera el modelo anterior (`del`, `gc.collect`, `empty_cache`) para que el pico no incluya restos de otro experimento. Sin CUDA se reporta N/A.
- **Tamaño del checkpoint:** full y probe con `trainer.save_model` (modelo completo, ~0,11 GB en fp32); LoRA con `save_pretrained` (solo adaptador + cabeza, pocos MB). Se mide con `dir_size_mb` (`os.walk`, como en NB2) y se muestra en MB y GB.
- **Tiempo:** reloj de pared (`time.perf_counter`) alrededor de `trainer.train()`.

### 13. Reproducibilidad y presentación

- Semilla 42 en `random`, `numpy`, `torch` y `transformers.set_seed` (también antes de construir cada modelo, para que la cabeza nueva se inicialice igual en los tres experimentos) y `seed=42` en `TrainingArguments`.
- Se imprimen las versiones de las librerías y el dict `HPARAMS` como tabla.
- Conclusiones impresas por código a partir de las métricas del run (ganador, deltas contra el probe, ratios de parámetros, tamaño, memoria y tiempo); el markdown solo contiene teoría y metodología.

