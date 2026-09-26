# 6. Pipeline de Random Forest

Código: `src/random_forest.py` (reutiliza `src/modelado.py`). Notebook: `notebooks/06_random_forest.ipynb`. Tablas y figuras con prefijo `06_` en `results/`.

## Conjuntos, referencia y métricas

Los mismos de la actividad 5: entrenamiento 2025 I, II y III (40,361 registros), validación 2025 IV (12,664 registros) y el primer trimestre de 2026 reservado para la evaluación final. La referencia es el salario medio de entrenamiento (Q3,383). MAE, RMSE y R² con `RegressionEvaluator`, sobre todos los registros de cada conjunto y sin ponderar. Residuo = real − predicho.

Para comparar, el notebook reajusta en los mismos conjuntos las configuraciones de la regresión lineal y vuelve a elegir la de menor RMSE de validación (es determinista, así que coincide con el notebook 05).

## Pipeline

1. `StringIndexer` (`handleInvalid="keep"`) y `OneHotEncoder` para nivel educativo, categoría ocupacional y dominio: las mismas etapas que la regresión (`modelado.etapas_preprocesamiento`).
2. `VectorAssembler` con edad, antigüedad, horas semanales y las categóricas codificadas.
3. `RandomForestRegressor` sobre `salario_mensual` en quetzales, **sin estandarizar** (un árbol divide por umbrales, y un cambio de escala solo mueve el umbral).

Parámetros fijos: `seed = config.SEMILLA` (42), `featureSubsetStrategy = "auto"` (un tercio de las columnas por división), `maxBins = 32`, `subsamplingRate = 1.0`.

## Configuraciones

| Configuración | `numTrees` | `maxDepth` |
|---|---|---|
| `arboles_20_prof_5` | 20 | 5 |
| `arboles_50_prof_5` | 50 | 5 |
| `arboles_50_prof_10` | 50 | 10 |
| `arboles_50_prof_15` | 50 | 15 |
| `arboles_50_prof_20` | 50 | 20 |
| `arboles_100_prof_20` | 100 | 20 |
| `arboles_50_prof_25` | 50 | 25 |

Se compara el número de árboles con la profundidad fija y la profundidad con 50 árboles fijos. La rejilla llega hasta 25 para comprobar dónde deja de mejorar la validación. Se elige la de **menor RMSE de validación**.

## Resultados en validación (2025 IV)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Referencia (media de entrenamiento) | Q1,673 | Q2,890 | −0.003 |
| Regresión lineal (`sin_regularizacion`) | Q1,210 | Q2,186 | 0.426 |
| Random Forest (`arboles_100_prof_20`) | **Q1,027** | **Q1,893** | **0.570** |

- La configuración elegida es **100 árboles con profundidad 20**. La profundidad es lo que más pesa: con 50 árboles el RMSE de validación baja de Q2,161 (profundidad 5) a Q1,986 (10) y Q1,896 (20), y se estanca en 25 (Q1,894). Duplicar los árboles cambia el RMSE en menos de Q5.
- El RMSE de validación de la configuración elegida es 9.4 % mayor que el de entrenamiento. La brecha crece con la profundidad, pero el error de validación no empeora en la rejilla: el bootstrap, el subconjunto de columnas por división y el promedio de 100 árboles contienen el sobreajuste.
- El Random Forest mejora a la regresión lineal en las tres métricas: −13.4 % de RMSE y −15.2 % de MAE, y el R² sube de 0.426 a 0.570.
- Importancia por predictor (suma de sus columnas): nivel educativo 34.3 %, categoría ocupacional 25.9 %, horas 12.4 %, edad 12.2 %, antigüedad 11.0 % y dominio 4.2 %.

## Error por tramo de salario (validación)

Con cortes en los percentiles 20, 40, 60 y 80 del salario real de validación:

| Tramo | n | MAE lineal | MAE RF | Error medio lineal | Error medio RF |
|---|---|---|---|---|---|
| ≤ Q1,600 | 2,606 | Q961 | Q746 | −Q830 | −Q672 |
| Q1,600 – Q2,800 | 2,676 | Q882 | Q693 | −Q408 | −Q353 |
| Q2,800 – Q3,600 | 2,600 | Q820 | Q693 | −Q118 | −Q97 |
| Q3,600 – Q4,400 | 2,261 | Q810 | Q712 | −Q99 | −Q64 |
| > Q4,400 | 2,521 | Q2,577 | Q2,297 | Q2,062 | Q1,718 |

Ambos modelos **sobreestiman los salarios bajos y subestiman los altos** (regresión hacia la media). El bosque tiene menor MAE en los cinco tramos, con la mayor ganancia en los extremos.

## ¿Por qué gana el Random Forest?

1. **No linealidad e interacciones.** El bosque puede representar que el efecto de edad, antigüedad u horas cambie según la educación o la categoría ocupacional. La regresión estima un efecto aditivo fijo por variable, y como su regularización casi no cambió el resultado, su límite es la forma del modelo, no el sobreajuste.
2. **Profundidad suficiente.** Con profundidad 5 el bosque apenas supera a la regresión (Q2,161 frente a Q2,186). La ventaja aparece con árboles profundos.
3. **Sobreajuste contenido** por el promedio de muchos árboles entrenados con muestras bootstrap.

## Archivos

- `results/tables/06_metricas_random_forest.csv`: métricas de entrenamiento y validación por configuración; marca la seleccionada.
- `results/tables/06_comparacion_validacion.csv`: referencia, mejor regresión lineal y mejor Random Forest en validación.
- `results/tables/06_importancias_random_forest.csv`: importancia de cada columna del vector de predictores.
- `results/tables/06_error_por_tramo_validacion.csv`: MAE y error medio por tramo de salario real, para ambos modelos.
- `results/figures/06_error_por_configuracion.png`, `06_comparacion_validacion.png`, `06_importancias.png` y `06_error_por_tramo.png`.

El mejor pipeline se guarda en `results/modelos/random_forest` (no se versiona). Para la actividad 7 se reajusta con los cuatro trimestres de 2025 usando `random_forest.pipeline_random_forest(100, 20)` con la misma semilla.

## Limitaciones

- Las importancias no tienen signo ni unidades y favorecen a las variables continuas; no son efectos causales.
- El bosque no extrapola fuera del rango de salarios de entrenamiento y encoge los extremos hacia el centro.
- Con seis predictores queda sin explicar cerca del 43 % de la variación del salario en validación.
- Las métricas describen los registros analizados, no a la población guatemalteca.
