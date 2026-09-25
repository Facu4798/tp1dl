# Smoke test — Ej. 3 (Swin-Tiny en EuroSAT)

- **Resultado:** PASS (nbconvert exit code 0).
- **Tiempo:** ~68 s de pared para todo el notebook (CPU, torch 2.14 CPU). Por experimento: probe ~6 s, full ~15 s, LoRA ~14 s.
- **Comando:** `JUPYTER_PATH=E:/Programs/miniconda/envs/tp1/share/jupyter TP1_SMOKE_TEST=1 TP1_LOCAL_DIR=C:/Users/facum/tp1_cache E:/Programs/miniconda/envs/tp1/python.exe -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 src/Adell_Giannice_Mazzola_Ej3_Swin_EuroSAT.ipynb --output-dir outputs/smoke --output ej3_smoke.ipynb`
- **Config smoke:** train 100 / val 50 / test 50 (estratificados), 1 época para probe, full y LoRA, batch 32, sin fp16.
- **Salida verificada en `outputs/smoke/ej3_smoke.ipynb`:** versiones e `HPARAMS`, grillas de ejemplos y de augmentation, distribución de clases, `print_trainable_parameters` (LoRA: 149.002 parámetros entrenables, 0,54%), curvas, tabla comparativa, barras, matriz de confusión + `classification_report`, errores y conclusiones impresas por código. Memoria de GPU: N/A (CPU).

## Errores encontrados y corregidos

1. **Kernel equivocado:** el kernelspec `python3` del usuario (`%APPDATA%/jupyter/kernels/python3`) apunta a `C:\Python314\python.exe` → `ModuleNotFoundError: datasets`. Solución sin tocar config global: anteponer `JUPYTER_PATH=E:/Programs/miniconda/envs/tp1/share/jupyter`.
2. **Nombres de atención de Swin en transformers 5.x:** son `q_proj`/`k_proj`/`v_proj`/`o_proj`, no `query`/`value`; `target_modules=['query','value']` da `NoMatchingPeftModuleError`. El notebook inspecciona `named_modules()` y resuelve `['query','value']` (transformers 4.x) o `['q_proj','v_proj']` (5.x).
3. **Recarga de checkpoints rota en transformers 5.17 para Swin:** `load_best_model_at_end` y `resume_from_checkpoint` cargaban el backbone con nombres legacy y `strict=False` (avisos de "missing/unexpected keys"), es decir, no lo cargaban. Se agregó `cargar_pesos` (recarga vía `from_pretrained`, `strict=True`) antes de reanudar y después de entrenar para probe/full. Se verificó aparte que con el fix los pesos quedan idénticos al checkpoint y sin él no.
4. `warmup_ratio` ya no existe en `TrainingArguments` 5.x → `warmup_steps=0.1` (fracción). `pin_memory` solo con CUDA.

No se instaló nada en el env `tp1`.
