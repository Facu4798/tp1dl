## Definition of done

- Los tres notebooks corren de punta a punta con `SMOKE_TEST=True` en el env local `tp1` (CPU), sin errores.
- Con `SMOKE_TEST=False` usan la configuración completa de la consigna (épocas, subsets, fp16 en Ej. 2/3, Drive, reanudación de checkpoints).
- Cada consigna de cada ejercicio tiene su celda/salida correspondiente: métricas, tablas, gráficos (curvas train/val, matrices de confusión, atención de B) y conclusiones impresas desde código; **ningún número de resultado escrito a mano en markdown**.
- El Ej. 1 no usa `nn.Transformer`, `nn.TransformerEncoder`, `nn.TransformerEncoderLayer` ni `nn.MultiheadAttention` (verificado con grep).
- Semilla, versiones de librerías e hiperparámetros impresos en cada notebook.
- Existe `docs/decisiones_de_diseno.md` con la justificación de todas las decisiones de los 3 ejercicios.
- La skill `code-style` fue aplicada sobre el código Python de los notebooks.
