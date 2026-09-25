## Constraints — Ejercicio 2 (NER, DistilBERT, CoNLL-2003)

- Modelo `distilbert-base-cased` con `AutoModelForTokenClassification`; dataset `eriktks/conll2003` cargado con `revision="refs/convert/parquet"`.
- Dataset pequeño → se usa completo (train / validation / test oficiales).
- Tokenizar con `is_split_into_words=True` y alinear etiquetas a subtokens (solo el primer subtoken de cada palabra lleva la etiqueta; el resto y los especiales `-100`); `DataCollatorForTokenClassification`.
- Métrica: F1 a nivel entidad con **seqeval** (vía `evaluate`) + accuracy (y precision/recall).
- LoRA: `TaskType.TOKEN_CLS`, target modules de atención de DistilBERT (`q_lin`, `v_lin`); la cabeza de clasificación entrenable también en LoRA (`modules_to_save=['classifier']`).
