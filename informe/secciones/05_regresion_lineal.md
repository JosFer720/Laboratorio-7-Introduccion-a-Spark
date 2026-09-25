# 5. Pipeline de regresión lineal

Código: `src/modelado.py` y `src/regresion_lineal.py`. Notebook: `notebooks/05_regresion_lineal.ipynb`. Tablas y figuras con prefijo `05_` en `results/`.

## Conjuntos

| Conjunto | Períodos | Uso |
|---|---|---|
| Entrenamiento | 2025 I, II y III | ajuste de todos los componentes del pipeline |
| Validación | 2025 IV | selección de la configuración |
| Prueba | 2026 I | reservado para la evaluación final; no se usa en esta sección |

`modelado.dividir_2025` falla si el conjunto de desarrollo trae algún período distinto de 2025, para que 2026 no se filtre por descuido.

## Modelo de referencia

Predice para todos los registros el **salario medio de entrenamiento**. Se reporta también la mediana de entrenamiento. Es el punto de comparación de la regresión lineal, del Random Forest y de la evaluación final.

## Pipeline

1. `StringIndexer` (`handleInvalid="keep"`) y `OneHotEncoder` para nivel educativo, categoría ocupacional y dominio.
2. `VectorAssembler` con edad, antigüedad, horas semanales y las categóricas codificadas (seis predictores).
3. Estandarización interna de `LinearRegression` (`standardization=True`); no se usa `StandardScaler`.
4. `LinearRegression` sobre `salario_mensual` en quetzales, sin transformar.

Las categorías que no aparecen en entrenamiento reciben una columna propia (siempre en cero al ajustar) en lugar de romper el pipeline en validación o prueba. `DESCONOCIDO` es una categoría más. Cada categoría observada tiene columna propia, sin nivel de referencia, por lo que los coeficientes categóricos se comparan entre categorías de una misma variable.

## Regularización

| Configuración | `regParam` | `elasticNetParam` |
|---|---|---|
| `sin_regularizacion` | 0.0 | 0.0 |
| `ridge_0.1` | 0.1 | 0.0 |
| `ridge_1.0` | 1.0 | 0.0 |
| `lasso_0.1` | 0.1 | 1.0 |
| `elasticnet_0.1_0.5` | 0.1 | 0.5 |

Cada una se ajusta solo con entrenamiento y se elige la de **menor RMSE de validación**.

## Métricas y residuo

MAE, RMSE y R² con `RegressionEvaluator`, sobre todos los registros de cada conjunto y sin ponderar. El residuo se define como **real − predicho**: positivo indica subestimación y negativo sobreestimación (`modelado.agregar_residuo`).

## Resultados

Las cifras finales se generan al ejecutar el notebook y quedan en:

- `results/tables/05_metricas_regresion_lineal.csv`: métricas de entrenamiento y validación por configuración; marca la seleccionada.
- `results/tables/05_coeficientes_regresion_lineal.csv`: intercepto y coeficientes del mejor modelo, en unidades originales.
- `results/figures/05_error_por_configuracion.png` y `results/figures/05_coeficientes.png`.

El mejor pipeline se guarda en `results/modelos/regresion_lineal` (no se versiona).

## Limitaciones

- El modelo es lineal y aditivo: no captura interacciones entre variables.
- Los salarios extremos se conservaron, por lo que influyen en el RMSE.
- Las asociaciones estimadas no son causales y las métricas describen los registros analizados, no a la población guatemalteca.
