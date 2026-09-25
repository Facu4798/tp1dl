## Constraints — verificación y proceso

- **Smoke test local**: cada notebook se ejecuta de punta a punta con `SMOKE_TEST=True` en CPU usando el entorno conda **`tp1`** (Python 3.11) de `E:/Programs/miniconda`, con torch CPU, torchaudio, torchvision, transformers, datasets, peft, evaluate, seqeval, accelerate, pandas, matplotlib, scikit-learn, jupyter/nbconvert. Ejecutar con `jupyter nbconvert --to notebook --execute` sobre una **copia** con `SMOKE_TEST=True` en `outputs/smoke/` (el notebook entregable en `src/` queda sin outputs y con `SMOKE_TEST=False`). Los resultados reales los generan los alumnos en Colab.
- El entorno `tp1` se crea una sola vez antes de trabajar en los notebooks (no en paralelo).
- El plan es **paralelizable por notebook**: cada notebook lo implementa un subagente independiente, sin el contexto de la conversación de planificación, que lee solo `CLAUDE.md`, su tarea de `PLAN.md`, sus `prompts/` y su `context/`.
- Verificar luego el código generado con la skill **`code-style`** (pasada final sobre el código Python).
- Limpieza final del planner: borrar solo `PROMPT.md`; **conservar `QUESTIONS.md`** en esta ocasión (y el documento de justificación).
