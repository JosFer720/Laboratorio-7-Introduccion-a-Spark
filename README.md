# Laboratorio 7 — Spark MLlib

Analisis exploratorio, segmentacion y regresion de salarios mensuales de personas asalariadas con la Encuesta Nacional de Empleo e Ingresos Continua (ENEIC, INE Guatemala), usando Python y Spark 3.5 (`pyspark.ml`).

- **Desarrollo:** cuatro trimestres de 2025.
- **Prueba final:** primer trimestre de 2026.
- **Poblacion analitica:** 15 anios o mas, ocupadas (`OCUPADOS = 1`), asalariadas (`P05C16` en 1-4) y con `P05D01` numerico, finito y positivo.
- **Objetivo:** `salario_mensual` (`P05D01`, ocupacion principal, en quetzales).

## Estructura

```
data/raw/          xlsx originales de la ENEIC (no se versionan)
data/processed/    parquet preparado de 2025 y 2026
informe/secciones/ secciones del informe
notebooks/         notebooks del laboratorio
results/           figuras, tablas y modelos
src/               modulos de carga, calidad, exploracion y segmentacion
tests/             pruebas con pytest
codebook.md        diccionario de las variables utilizadas
```

## Requisitos

- Python 3.10+ y Spark 3.5.x (`pyspark==3.5.*`).
- Java 8, 11, 17 o 21 disponible en el `PATH`.

```bash
pip install -r requirements.txt
```

Colocar los xlsx de Personas en `data/raw/` con los nombres indicados en `src/carga.py` (`ARCHIVOS`).

## Ejecucion

```bash
jupyter notebook notebooks/
pytest
```

Los notebooks se ejecutan de principio a fin sin variables previas. La semilla global es `SEMILLA` en `src/config.py`.

## Notas metodologicas

- Clustering, modelos y metricas son **no ponderados**; los resultados describen los registros analizados y no son estimaciones oficiales de la poblacion. `FACTOR` se conserva para un analisis poblacional, donde se usaria como peso de expansion.
- La columna `TRIMESTRE` original no es el trimestre calendario; se usa `periodo_archivo`, derivado del archivo de procedencia.
- Las asociaciones no implican causalidad.

## Estado

| Seccion | Estado |
|---|---|
| Carga, armonizacion y calidad | implementada (src/carga.py, src/calidad.py, notebook 01) |
| Estadistica descriptiva | pendiente |
| Correlaciones | pendiente |
| Segmentacion KMeans | pendiente |
| Regresion lineal, Random Forest, evaluacion 2026 y errores | pendiente |
