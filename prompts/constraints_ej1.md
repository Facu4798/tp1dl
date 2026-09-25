## Constraints — Ejercicio 1 (ESC-50: CNN 2D vs. Transformer A/B)

- Librerías: PyTorch + **torchaudio** (carga y mel-spectrogram). **fp32** está bien (no hace falta precisión mixta).
- **Prohibido** usar `torch.nn.Transformer`, `torch.nn.TransformerEncoder`, `torch.nn.TransformerEncoderLayer` o `torch.nn.MultiheadAttention`. La atención escalada, las cabezas múltiples, el feed-forward y la normalización se programan explícitamente (se puede partir de los bloques de las prácticas de la Unidad II). Tokenizar el espectrograma en parches, proyectar cada token linealmente a d_model, sumar la codificación posicional y pasarlo por los bloques de atención.
- **Split**: folds 1–3 train / fold 4 validación (early stopping) / fold 5 test. Dataset pequeño → se usa completo.
- **Representación**: resamplear a 22050 Hz, mel-spectrogram con n_mels=128 en dB (≈128 × ~216 frames para 5 s). Describir en markdown los parámetros de ventana (n_fft, hop, ventana Hann) y el trade-off tiempo-frecuencia; mostrar señales en el tiempo y su mel-spectrogram.
- **Épocas**: orden de 20–50 con early stopping sobre validación.
- **Tres modelos**:
  - **CNN 2D** sobre el mel-espectrograma, con SpecAugment en train.
  - **Config A**: tokenización en secuencias cortas ≈12 tokens = **franjas temporales de banda completa** (128 mel × ~18 frames cada una), 2 capas, d_model 64, 2 cabezas, **sin dropout**, **post-norm**, **sin SpecAugment** (sin ninguna regularización).
  - **Config B**: tokenización en secuencias largas (~50–200 tokens) = **parches 2D estilo AST**, 2–3 capas, d_model 64–128, 2–4 cabezas, dropout 0.1–0.2, **pre-norm** con **warmup** del LR, con SpecAugment.
  - A y B son el **mismo encoder** (misma clase, parametrizada) y de tamaño comparable ("mismo tamaño"); difieren en la tokenización y la regularización.
- **Reportar para los tres**: accuracy y F1-macro en test, matriz de confusión, número de parámetros y tiempo de entrenamiento (tabla comparativa), más curvas train/val.
- **Visualizar los pesos de atención de la Config B** sobre la señal/espectrograma (mapear la atención por parche de vuelta a la grilla del espectrograma y superponerla).
