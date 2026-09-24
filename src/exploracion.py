"""Estadistica descriptiva, agregaciones para graficar y correlaciones."""
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src import config

VARIABLES_NUMERICAS = ["salario_mensual", "edad", "antiguedad", "horas_semanales"]
ETIQUETAS = {
    "salario_mensual": "Salario mensual (Q)",
    "edad": "Edad (años)",
    "antiguedad": "Antigüedad (años)",
    "horas_semanales": "Horas semanales",
}
PERCENTILES = [0.25, 0.5, 0.75, 0.95]


def descriptivas(df: DataFrame, columnas: list = None) -> pd.DataFrame:
    """n, media, mediana, desv. estandar, minimo, maximo, P25, P75 y P95 sobre todo el conjunto.

    Los percentiles son exactos (no aproximados) y se calculan con todos los registros.
    """
    columnas = columnas or VARIABLES_NUMERICAS
    exprs = []
    for c in columnas:
        exprs += [
            F.count(c).alias(f"{c}|n"),
            F.mean(c).alias(f"{c}|media"),
            F.stddev(c).alias(f"{c}|desv_estandar"),
            F.min(c).alias(f"{c}|minimo"),
            F.max(c).alias(f"{c}|maximo"),
            F.expr(f"percentile({c}, array({', '.join(map(str, PERCENTILES))}))").alias(f"{c}|pct"),
        ]
    fila = df.agg(*exprs).first()
    filas = []
    for c in columnas:
        p25, p50, p75, p95 = fila[f"{c}|pct"]
        filas.append({
            "variable": c, "n": fila[f"{c}|n"], "media": fila[f"{c}|media"], "mediana": p50,
            "desv_estandar": fila[f"{c}|desv_estandar"], "minimo": fila[f"{c}|minimo"],
            "maximo": fila[f"{c}|maximo"], "p25": p25, "p75": p75, "p95": p95,
        })
    return pd.DataFrame(filas).set_index("variable")


def asimetria(df: DataFrame, columna: str) -> dict:
    """Media, mediana y coeficiente de asimetria (skewness) de una variable."""
    fila = df.agg(F.mean(columna).alias("media"),
                  F.expr(f"percentile({columna}, 0.5)").alias("mediana"),
                  F.skewness(columna).alias("asimetria")).first()
    return {"media": fila["media"], "mediana": fila["mediana"], "asimetria": fila["asimetria"]}


def conteo_categorias(df: DataFrame, columna: str) -> pd.DataFrame:
    """Registros y porcentaje por categoria, ordenado por codigo."""
    total = df.count()
    tabla = df.groupBy(columna).count().toPandas().rename(columns={columna: "categoria", "count": "registros"})
    tabla["porcentaje"] = (100 * tabla["registros"] / total).round(2)
    return tabla.sort_values("categoria", key=_orden_codigo, ignore_index=True)


def _orden_codigo(serie: pd.Series) -> pd.Series:
    """Ordena codigos numericos como numeros y deja DESCONOCIDO al final."""
    return pd.to_numeric(serie, errors="coerce").fillna(float("inf"))


def salario_por_grupo(df: DataFrame, columna: str) -> pd.DataFrame:
    """Registros, media y mediana del salario por categoria de `columna`."""
    tabla = (df.groupBy(columna)
               .agg(F.count(F.lit(1)).alias("registros"),
                    F.mean("salario_mensual").alias("salario_medio"),
                    F.expr("percentile(salario_mensual, 0.5)").alias("salario_mediano"))
               .toPandas().rename(columns={columna: "categoria"}))
    return tabla.sort_values("categoria", key=_orden_codigo, ignore_index=True)


def resumen_por_trimestre(df: DataFrame) -> pd.DataFrame:
    """Tamano de la muestra analitica y salario mediano por periodo de archivo."""
    return (df.groupBy("periodo_archivo")
              .agg(F.count(F.lit(1)).alias("registros"),
                   F.expr("percentile(salario_mensual, 0.5)").alias("salario_mediano"),
                   F.mean("salario_mensual").alias("salario_medio"))
              .orderBy("periodo_archivo").toPandas())


def histograma(df: DataFrame, columna: str, bins: int = 40, log10: bool = False) -> pd.DataFrame:
    """Conteos por intervalo calculados en Spark (solo el agregado pasa a pandas).

    Con `log10=True` los intervalos se construyen sobre log10(valor); las columnas
    `desde` y `hasta` quedan en la escala logaritmica.
    """
    valor = F.log10(F.col(columna)) if log10 else F.col(columna)
    base = df.select(valor.alias("v")).filter(F.col("v").isNotNull() & ~F.isnan("v"))
    minimo, maximo = base.agg(F.min("v"), F.max("v")).first()
    ancho = (maximo - minimo) / bins or 1.0
    conteos = (base.select(F.least(F.floor((F.col("v") - minimo) / ancho), F.lit(bins - 1)).alias("bin"))
                   .groupBy("bin").count().toPandas().set_index("bin")["count"])
    tabla = pd.DataFrame({"bin": range(bins)})
    tabla["desde"] = minimo + tabla["bin"] * ancho
    tabla["hasta"] = tabla["desde"] + ancho
    tabla["registros"] = tabla["bin"].map(conteos).fillna(0).astype(int)
    return tabla.drop(columns="bin")


def muestra(df: DataFrame, n: int = 5000, semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Muestra aleatoria de hasta `n` registros para graficar (nunca para calcular metricas)."""
    total = df.count()
    fraccion = min(1.0, 1.2 * n / total) if total else 1.0
    return df.sample(fraction=fraccion, seed=semilla).limit(n).toPandas()


def matriz_correlacion(df: DataFrame, columnas: list = None) -> pd.DataFrame:
    """Correlacion de Pearson con VectorAssembler + Correlation.corr sobre todos los registros."""
    columnas = columnas or VARIABLES_NUMERICAS
    vectores = VectorAssembler(inputCols=columnas, outputCol="x", handleInvalid="skip").transform(df)
    matriz = Correlation.corr(vectores, "x", "pearson").head()[0].toArray()
    return pd.DataFrame(matriz, index=columnas, columns=columnas)
