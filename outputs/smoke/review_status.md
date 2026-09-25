# Task 12 — Revisión cruzada y pasada de code-style

Fecha: 2026-09-25 · Estado: **PASS**

## Skill code-style (sobre `src/_build/build_ej{1,2,3}.py`)
- Ej. 1:
  - Quitado el try/except del montaje de Drive (en Colab se monta directo; fuera de Colab se usa `TP1_LOCAL_DIR`).
  - Quitados tres `assert` redundantes: el del cache de mels, `d_model % n_heads` y el `isinstance` de clases prohibidas (esto último lo verifica el grep de abajo).
  - Quitado el fallback de ejemplos de atención.
- Ej. 2:
  - Quitado el try/except del montaje de Drive.
  - Quitado el try/except de los nombres de etiquetas (el `ClassLabel` está verificado en el dataset parquet).
  - Quitado el `isinstance` sobre `p.predictions`.
  - `docs/decisiones_de_diseno.md` actualizado en consecuencia.
- Ej. 3: quitado el try/except del montaje de Drive.

## Verificación técnica adicional
- Recarga de checkpoints en transformers 5.17:
  - DistilBERT guarda con las mismas claves que su `state_dict`, así que el `load_best_model_at_end` del Ej. 2 es correcto.
  - Swin guarda 156 claves con nombres legacy que no coinciden con el modelo, así que el workaround `cargar_pesos` del Ej. 3 es necesario.

## Checks
| Check | Resultado |
|---|---|
| grep `nn.(Transformer\|TransformerEncoder\|TransformerEncoderLayer\|MultiheadAttention)` en el notebook del Ej. 1 | sin coincidencias |
| `nbformat.validate` de los 3 notebooks de `src/` | OK |
| Outputs en los notebooks de `src/` | 0 en los tres |
| Resultados numéricos en markdown (regex de decimales, % y acc/F1) | sin coincidencias |
| Smoke test Ej. 1 (re-run) | exit 0, 15 s, 0 celdas con error |
| Smoke test Ej. 2 (re-run) | exit 0, 39 s, 0 celdas con error |
| Smoke test Ej. 3 (re-run) | exit 0, 72 s, 0 celdas con error |

## Consistencia entre notebooks
- Los tres siguen el orden de la celda de setup de `CLAUDE.md`: Colab, `SMOKE_TEST`, `BASE`, semilla/device, versiones y `HPARAMS`.
- Diferencia aceptada: el Ej. 2 agrega `EarlyStoppingCallback(patience=1)` y el Ej. 3 solo `load_best_model_at_end`. Con 3 épocas, en ambos casos se evalúa el mejor checkpoint de validación. Está documentado en `docs/decisiones_de_diseno.md`.
