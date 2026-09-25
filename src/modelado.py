"""Base del modelado supervisado: conjuntos, modelo de referencia y metricas de regresion."""
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src import config

OBJETIVO = "salario_mensual"
PREDICTORES = config.NUMERICAS + config.CATEGORICAS
PERIODOS_ENTRENAMIENTO = config.PERIODOS_TRAIN[:3]
PERIODO_VALIDACION = config.PERIODOS_TRAIN[3]
PREDICCION = "prediccion"
RESIDUO = "residuo"


def leer_preparado(spark: SparkSession, nombre: str) -> DataFrame:
    """Lee un conjunto preparado (personas_2025 / personas_2026) desde data/processed."""
    return spark.read.parquet(str(config.DATA_PROCESSED / f"{nombre}.parquet"))


def dividir_2025(df: DataFrame) -> tuple:
    """Entrenamiento (2025T1-T3) y validacion (2025T4). Falla si el conjunto trae otros periodos."""
    periodos = {r[0] for r in df.select("periodo_archivo").distinct().collect()}
    fuera = periodos - set(config.PERIODOS_TRAIN)
    if fuera:
        raise ValueError(f"El conjunto de desarrollo solo puede contener 2025; sobran: {sorted(fuera)}")
    entrenamiento = df.filter(F.col("periodo_archivo").isin(PERIODOS_ENTRENAMIENTO))
    validacion = df.filter(F.col("periodo_archivo") == PERIODO_VALIDACION)
    return entrenamiento, validacion


def cargar_conjuntos(spark: SparkSession) -> tuple:
    """(entrenamiento, validacion) a partir de personas_2025.parquet."""
    return dividir_2025(leer_preparado(spark, "personas_2025"))


def cargar_prueba(spark: SparkSession) -> DataFrame:
    """Conjunto de prueba final (2026T1). Solo se usa en la evaluacion final."""
    return leer_preparado(spark, "personas_2026")


def referencia(entrenamiento: DataFrame, estadistico: str = "media") -> float:
    """Valor constante del modelo de referencia, calculado solo con entrenamiento."""
    if estadistico == "media":
        return float(entrenamiento.agg(F.avg(OBJETIVO)).first()[0])
    if estadistico == "mediana":
        return float(entrenamiento.approxQuantile(OBJETIVO, [0.5], 0.001)[0])
    raise ValueError("estadistico debe ser 'media' o 'mediana'")


def predecir_referencia(df: DataFrame, valor: float) -> DataFrame:
    """Agrega la prediccion constante del modelo de referencia."""
    return df.withColumn(PREDICCION, F.lit(float(valor)))


def metricas(pred: DataFrame, etiqueta: str = OBJETIVO, prediccion: str = PREDICCION) -> dict:
    """MAE, RMSE y R2 sobre todos los registros de `pred` (sin ponderar)."""
    resultado = {
        m: RegressionEvaluator(labelCol=etiqueta, predictionCol=prediccion, metricName=m).evaluate(pred)
        for m in ("mae", "rmse", "r2")
    }
    resultado["n"] = pred.count()
    return resultado


def agregar_residuo(pred: DataFrame, etiqueta: str = OBJETIVO, prediccion: str = PREDICCION) -> DataFrame:
    """Residuo = real - predicho: positivo indica subestimacion, negativo sobreestimacion."""
    return pred.withColumn(RESIDUO, F.col(etiqueta) - F.col(prediccion))


def tabla_metricas(filas: dict) -> pd.DataFrame:
    """{nombre: dict de metricas} -> tabla con una fila por modelo."""
    tabla = pd.DataFrame(filas).T
    tabla.index.name = "modelo"
    return tabla.reset_index()
