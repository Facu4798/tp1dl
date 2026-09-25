## Constraints — Ejercicio 3 (Swin-Tiny, EuroSAT)

- Modelo `microsoft/swin-tiny-patch4-window7-224` con `AutoModelForImageClassification` (`ignore_mismatched_sizes=True`, `num_labels=10`, `id2label/label2id`); dataset `timm/eurosat-rgb`.
- Cargar con su **image processor** (`AutoImageProcessor`: resize a 224, normalización) y **augmentation básica** en train (p. ej. flips / rotaciones leves con torchvision), aplicada on-the-fly con `set_transform`/`with_transform`.
- **Subset fijo de 10k imágenes de train, estratificado por clase, con semilla fija**, usado **igual para full y LoRA** (mismas condiciones). Validación y test = splits oficiales de `timm/eurosat-rgb`.
- LoRA: `target_modules=['query','value']` (atención de Swin); la cabeza (`classifier`) entrenable vía `modules_to_save`.
- Métrica: accuracy (y F1-macro); reportar también memoria de GPU.
