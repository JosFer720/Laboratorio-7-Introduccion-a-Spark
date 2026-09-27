# 8. Visualización y análisis de errores

Código: `src/evaluacion.py`. Notebook: `notebooks/08_analisis_errores.ipynb`. Tablas y figuras con prefijo `08_` en `results/`.

## Convenciones

- **Residuo = real − predicho.** Positivo = subestimación; negativo = sobreestimación.
- Los gráficos de dispersión usan **la misma muestra** de hasta 5,000 registros de prueba para los dos modelos (`evaluacion.muestra_comun`, semilla 42), solo para dibujar.
- Las métricas por grupo, por tramo de salario y por decil de predicción se calculan en Spark con **todos** los registros de prueba (13,258).

## Distribución del salario

| | Media | Desv. est. | P25 | Mediana | P75 | P90 | P95 | P99 | Máximo |
|---|---|---|---|---|---|---|---|---|---|
| 2025 | Q3,422 | Q2,902 | Q1,800 | Q3,000 | Q4,000 | Q6,000 | Q8,000 | Q15,000 | Q99,000 |
| 2026 I | Q3,566 | Q2,868 | Q2,000 | Q3,200 | Q4,066 | Q6,000 | Q8,000 | Q15,000 | Q60,000 |

- El salario es muy asimétrico a la derecha (asimetría 6.0 en 2025 y 4.4 en 2026), y el último 10 % se aleja mucho del centro.
- 2026 es algo más alto que 2025 en la parte baja y media, mientras que la cola alta coincide.
- Muchos salarios se reportan en cifras redondas.

## Salario real frente a predicho, y residuos

- En los dos modelos la nube es más plana que la línea y = x. El P99 de lo predicho (Q10,948 la regresión y Q10,596 el bosque) queda muy por debajo del P99 real (Q15,000).
- La **regresión** predice **29 salarios negativos** (mínimo −Q496). El bosque nunca sale del rango observado en entrenamiento.
- Las personas con maestría o doctorado forman un grupo separado de predicciones cercanas a Q11,000. Como sus salarios reales son muy dispersos, en ese grupo hay a la vez grandes sobreestimaciones y grandes subestimaciones.
- **Heterocedasticidad:** el MAE sube de Q596 (regresión) y Q400 (bosque) en el decil más bajo de predicción a Q3,250 y Q3,029 en el más alto. El residuo está acotado hacia abajo (el salario real es positivo), pero hacia arriba llega a decenas de miles de quetzales.
- En los deciles de predicción 3.º a 9.º, el error medio queda cerca de cero. En el decil más alto los dos modelos subestiman en promedio unos Q600 a Q700. En los dos deciles más bajos, la regresión subestima Q318 y Q261, mientras que el bosque queda cerca de cero.

## Error por nivel educativo y por dominio (todos los registros de prueba)

| Nivel educativo | n | Salario real medio | MAE lineal | MAE RF | Error medio lineal | Error medio RF |
|---|---|---|---|---|---|---|
| Ninguno | 1,016 | Q1,884 | Q824 | Q605 | Q109 | Q63 |
| Preprimaria | 80 | Q2,375 | Q735 | Q535 | Q89 | Q57 |
| Primaria | 3,957 | Q2,423 | Q870 | Q737 | Q55 | Q57 |
| Básico | 2,164 | Q2,905 | Q894 | Q776 | Q128 | Q124 |
| Diversificado | 4,117 | Q3,924 | Q1,109 | Q986 | Q104 | Q96 |
| Superior | 1,729 | Q6,264 | Q2,566 | Q2,226 | Q296 | Q267 |
| Maestría | 183 | Q11,291 | Q5,553 | Q4,523 | −Q132 | −Q325 |
| Doctorado | 12 | Q20,292 | Q12,919 | Q9,430 | Q6,063 | Q4,812 |

| Dominio | n | Salario real medio | MAE lineal | MAE RF | Error medio lineal | Error medio RF |
|---|---|---|---|---|---|---|
| Urbano metropolitano | 5,632 | Q4,352 | Q1,446 | Q1,282 | Q136 | Q145 |
| Resto urbano | 4,846 | Q3,282 | Q1,190 | Q998 | Q134 | Q93 |
| Rural nacional | 2,780 | Q2,468 | Q914 | Q735 | Q65 | Q53 |

- El MAE crece con la educación y es mayor en el dominio urbano metropolitano, donde también son más altos y más dispersos los salarios. El Random Forest tiene menor MAE en todos los niveles educativos y en los tres dominios.
- El error medio es positivo en casi todos los grupos, con una subestimación promedio de unos Q50 a Q300, coherente con el aumento del salario de 2025 a 2026. Doctorado tiene solo 12 registros, así que su cifra no es concluyente.

## Error por percentil del salario real

Cortes en P25, P50, P75, P90 y P95 de 2026:

| Tramo | n | Salario real medio | Predicción media lineal | Predicción media RF | Error medio lineal | Error medio RF |
|---|---|---|---|---|---|---|
| ≤ Q2,000 | 4,032 | Q1,298 | Q2,126 | Q1,997 | −Q829 | −Q700 |
| Q2,000 – Q3,200 | 2,715 | Q2,738 | Q2,899 | Q2,871 | −Q161 | −Q133 |
| Q3,200 – Q4,066 | 3,188 | Q3,769 | Q3,849 | Q3,829 | −Q80 | −Q59 |
| Q4,066 – Q6,000 | 2,072 | Q4,889 | Q4,402 | Q4,441 | Q486 | Q447 |
| Q6,000 – Q8,000 | 608 | Q7,225 | Q5,591 | Q5,815 | Q1,634 | Q1,410 |
| > Q8,000 | 643 | Q12,553 | Q6,907 | Q7,880 | Q5,645 | Q4,673 |

| | Regresión lineal | Random Forest |
|---|---|---|
| Mediana del error absoluto | Q801 | Q639 |
| Registros con error ≤ 20 % del salario real | 38 % | 46 % |
| Subestimados, toda la prueba | 49 % | 48 % |
| Subestimados, salario real > P90 | **92 %** | **89 %** |
| Parte del error cuadrático que aporta el 5 % con salario real > P95 | **62 %** | **59 %** |

- Los errores son grandes frente al salario típico.
- Ambos modelos **sobreestiman los salarios bajos y subestiman sistemáticamente los altos** (regresión hacia la media).
- La cola alta domina el RMSE. Como la evaluación principal conserva los salarios extremos sin recortes ni transformaciones, el RMSE refleja sobre todo esa cola.
- Los salarios muy altos dependen de factores que no están entre los seis predictores (ocupación, rama, empresa, cargo).

## Discusión final (resumen)

1. **Perfiles (pregunta 1).** Con edad, antigüedad y horas se distinguen cuatro perfiles de trayectoria y jornada. Solo el de trayectoria larga tiene un salario mediano mayor, así que esas variables, por sí solas, separan poco a los trabajadores por salario.
2. **Estimación del salario (pregunta 2).** Con seis predictores, el Random Forest explica 57 % de la variación del salario en 2026 (la regresión, 43 %), con un desempeño estable entre validación y prueba. Gana porque capta no linealidades e interacciones. La regresión está limitada por su forma aditiva y produce algunos salarios negativos.
3. **Utilidad.** La estimación sirve para describir el salario típico de un perfil, no para predecir el de una persona concreta: el MAE equivale a un tercio del salario mediano y los salarios altos se subestiman de forma sistemática.
4. **Limitaciones.** El análisis no es ponderado (para estimaciones poblacionales se usaría `FACTOR`), no es causal y le faltan predictores clave. Además, el modelo entrenado con 2025 subestima levemente el nivel de 2026. Un análisis complementario podría modelar el logaritmo del salario o usar una pérdida robusta, sin reemplazar la comparación obligatoria.

## Archivos

- `results/tables/08_percentiles_salario.csv`, `08_rango_predicciones.csv`, `08_error_por_decil_prediccion.csv`, `08_error_por_nivel_educativo.csv`, `08_error_por_dominio.csv`, `08_error_por_tramo_prueba.csv` y `08_resumen_errores.csv`.
- `results/figures/08_percentiles_salario.png`, `08_real_vs_predicho.png`, `08_residuos_vs_predicho.png`, `08_error_por_grupo.png` y `08_error_por_tramo.png`.
