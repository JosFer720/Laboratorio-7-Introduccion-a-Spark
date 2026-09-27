# 2 y 3. Estadística descriptiva, exploración y correlaciones

Código: `src/exploracion.py`. Notebook: `notebooks/02_exploracion_y_correlaciones.ipynb`. Tablas y figuras con prefijo `02_` en `results/`.

Todas las estadísticas y correlaciones se calculan en Spark con los 53,025 registros de 2025. A pandas solo pasan agregados.

## Estadística descriptiva (2025)

| Variable | n | Media | Mediana | Desv. est. | Mín. | Máx. | P25 | P75 | P95 |
|---|---|---|---|---|---|---|---|---|---|
| Salario mensual (Q) | 53,025 | 3,422 | 3,000 | 2,902 | 1 | 99,000 | 1,800 | 4,000 | 8,000 |
| Edad (años) | 53,025 | 35.2 | 33 | 13.5 | 15 | 89 | 24 | 44 | 61 |
| Antigüedad (años) | 53,025 | 5.6 | 2.0 | 7.8 | 0 | 70 | 0.5 | 7 | 22 |
| Horas semanales | 53,025 | 46.3 | 45 | 18.6 | 1 | 126 | 40 | 55 | 80 |

## Preguntas de exploración

- **Distribución por grupos.** Empresa privada 54.1 %, jornaleros o peones 26.1 %, gobierno 12.4 % y servicio doméstico 7.4 %. En educación, diversificado (31.8 %) y primaria (29.8 %) concentran más del 60 %, mientras que maestría (1.5 %) y doctorado (0.1 %, 60 registros) son grupos muy pequeños. Por dominio: urbano metropolitano 41.6 %, resto urbano 38.0 % y rural nacional 20.4 %.
- **Forma del salario.** Asimétrica a la derecha (coeficiente de asimetría 6.0). En escala log10, que se usa solo para visualizar y está rotulada, la distribución se parece a una campana alrededor de Q3,000.
- **Media frente a mediana.** La media (Q3,422) supera a la mediana (Q3,000) en Q422 (+14.1 %) por la cola de salarios altos. Los extremos no se eliminaron.
- **Salario mediano por grupo.** Crece con la educación: Q1,500 sin educación, Q2,200 con primaria, Q3,600 con diversificado, Q5,000 con superior y de Q10,000 a Q12,000 con maestría o doctorado. Por categoría: gobierno Q5,000, empresa privada Q3,568, jornaleros Q1,800 y servicio doméstico Q1,000.
- **Trimestres.** El tamaño de la muestra es estable (de 12,664 a 13,492). El salario mediano pasa de Q3,000 (I y II) a Q3,200 (III y IV). Es un aumento pequeño, no concluyente.

## Correlaciones de Pearson (`VectorAssembler` + `Correlation.corr`)

| | Salario | Edad | Antigüedad | Horas |
|---|---|---|---|---|
| Salario | 1.00 | 0.15 | 0.18 | 0.08 |
| Edad | 0.15 | 1.00 | 0.49 | −0.11 |
| Antigüedad | 0.18 | 0.49 | 1.00 | −0.06 |
| Horas | 0.08 | −0.11 | −0.06 | 1.00 |

- La mayor asociación lineal con el salario es la de la antigüedad (r = 0.18), seguida de la edad (0.15). Las tres son débiles: la antigüedad sola explica linealmente 3.2 % de la variación del salario.
- Edad y antigüedad tienen una relación positiva y moderada (r = 0.49), pero no tan alta como para que una sustituya a la otra.
- Pearson solo capta relaciones lineales y es sensible a los extremos. Las correlaciones no implican causalidad, y las observaciones repetidas de una misma persona no son independientes.

## Archivos

- `results/tables/02_descriptivas_2025.csv`, `02_salario_por_grupo_2025.csv`, `02_resumen_trimestre_2025.csv` y `02_correlacion_2025.csv`.
- `results/figures/02_distribucion_categorias.png`, `02_histograma_salario.png`, `02_salario_mediano_grupos.png`, `02_muestra_y_mediana_trimestre.png` y `02_mapa_calor_correlacion.png`.
