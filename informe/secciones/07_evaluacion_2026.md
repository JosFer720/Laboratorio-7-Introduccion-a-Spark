# 7. Entrenamiento final y evaluación en 2026

Código: `src/evaluacion.py` (reutiliza `src/modelado.py`, `src/regresion_lineal.py` y `src/random_forest.py`). Notebook: `notebooks/07_evaluacion_final_2026.ipynb`. Tablas y figuras con prefijo `07_` en `results/`.

## Configuraciones y reentrenamiento

Se toma de cada algoritmo la configuración con menor RMSE de validación (2025 IV), leída de `05_metricas_regresion_lineal.csv` y `06_metricas_random_forest.csv`:

| Algoritmo | Configuración | Hiperparámetros | RMSE de validación |
|---|---|---|---|
| Regresión lineal | `sin_regularizacion` | `regParam = 0.0`, `elasticNetParam = 0.0` | Q2,186 |
| Random Forest | `arboles_100_prof_20` | `numTrees = 100`, `maxDepth = 20`, `seed = 42` | Q1,893 |

Cada pipeline completo (indexadores, codificador, estandarización interna de la regresión y modelo) se **reajusta con los cuatro trimestres de 2025** (53,025 registros). El trimestre IV ya no hace falta como validación, porque la configuración quedó fijada con él, y así el modelo final aprovecha más registros y el período más cercano a 2026. `evaluacion.entrenar_finales` falla si el conjunto trae períodos que no son de 2025.

La referencia se recalcula con el mismo conjunto: salario medio de 2025, **Q3,422** (mediana Q3,000).

## Prueba: 2026 I

- `personas_2026.parquet` se preparó con las mismas funciones y reglas que 2025 (`calidad.preparar`): 13,258 registros elegibles.
- `evaluacion.predecir` aplica la referencia, la regresión y el bosque uno tras otro sobre el mismo DataFrame, sin uniones. Así, **los dos modelos se evalúan sobre exactamente los mismos registros**. Se verificó que las 13,258 filas tienen las tres predicciones, sin nulos, y que cada clave aparece una sola vez.
- No hay categorías de 2026 que no existieran en 2025.

## Resultados en prueba

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Referencia (media de 2025) | Q1,718 | Q2,871 | −0.003 |
| Regresión lineal | Q1,241 | Q2,164 | 0.431 |
| Random Forest | **Q1,064** | **Q1,891** | **0.565** |

- Frente a la referencia, la regresión baja el RMSE 24.6 % y el Random Forest 34.1 %.
- El **Random Forest** es el mejor en las tres métricas. Frente a la regresión, reduce el RMSE 12.6 % y el MAE 14.3 %.
- El MAE del mejor modelo equivale a 35 % del salario mediano, y el RMSE es 1.8 veces el MAE: unos pocos errores muy grandes pesan mucho (actividad 8).

## Validación frente a prueba

| Modelo | RMSE validación | RMSE prueba | Cambio | R² validación | R² prueba |
|---|---|---|---|---|---|
| Regresión lineal | Q2,186 | Q2,164 | −1.0 % | 0.426 | 0.431 |
| Random Forest | Q1,893 | Q1,891 | −0.1 % | 0.570 | 0.565 |

Los dos modelos **generalizan a 2026**: el desempeño casi no cambia y el orden (Random Forest < regresión lineal < referencia) se mantiene. El MAE sube entre 2.5 % y 3.6 %, en parte porque los salarios de 2026 son algo más altos (media Q3,566 y mediana Q3,200, frente a Q3,422 y Q3,000 en 2025).

## Archivos

- `results/tables/07_configuraciones_finales.csv`, `07_metricas_prueba_2026.csv` y `07_validacion_vs_prueba.csv`.
- `results/figures/07_metricas_prueba.png` y `07_validacion_vs_prueba.png`.
- Los pipelines finales se guardan en `results/modelos/regresion_lineal_final` y `random_forest_final` (no se versionan).
