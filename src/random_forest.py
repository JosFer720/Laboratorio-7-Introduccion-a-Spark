"""Pipeline de Random Forest para estimar salario_mensual."""
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src import config, modelado

# Por defecto para regresion: cada division considera un tercio de los predictores ("onethird").
ESTRATEGIA_SUBCONJUNTO = "auto"
MAX_BINS = 32
# nombre -> (numTrees, maxDepth). Se varia la cantidad de arboles y la profundidad maxima.
CONFIGURACIONES = {
    "arboles_20_prof_5": (20, 5),
    "arboles_50_prof_5": (50, 5),
    "arboles_50_prof_10": (50, 10),
    "arboles_50_prof_15": (50, 15),
    "arboles_50_prof_20": (50, 20),
    "arboles_100_prof_20": (100, 20),
    "arboles_50_prof_25": (50, 25),
}
# Cortes (percentiles) para resumir el error por tramo de salario real.
PERCENTILES_TRAMOS = [0.2, 0.4, 0.6, 0.8]


def pipeline_random_forest(num_trees: int, max_depth: int, semilla: int = config.SEMILLA) -> Pipeline:
    """Preprocesamiento categorico + RandomForestRegressor (sin estandarizar: los arboles no la necesitan)."""
    bosque = RandomForestRegressor(
        featuresCol="features", labelCol=modelado.OBJETIVO, predictionCol=modelado.PREDICCION,
        numTrees=num_trees, maxDepth=max_depth, maxBins=MAX_BINS,
        featureSubsetStrategy=ESTRATEGIA_SUBCONJUNTO, subsamplingRate=1.0, seed=semilla)
    return Pipeline(stages=modelado.etapas_preprocesamiento() + [bosque])


def evaluar_configuraciones(entrenamiento: DataFrame, validacion: DataFrame,
                            configuraciones: dict = None) -> tuple:
    """Ajusta cada configuracion solo con entrenamiento y la evalua en entrenamiento y validacion.

    Devuelve (tabla de metricas, {nombre: PipelineModel}).
    """
    configuraciones = configuraciones or CONFIGURACIONES
    filas, modelos = [], {}
    for nombre, (num_trees, max_depth) in configuraciones.items():
        modelo = pipeline_random_forest(num_trees, max_depth).fit(entrenamiento)
        m_train = modelado.metricas(modelo.transform(entrenamiento))
        m_valid = modelado.metricas(modelo.transform(validacion))
        filas.append({
            "configuracion": nombre, "numTrees": num_trees, "maxDepth": max_depth,
            "mae_entrenamiento": m_train["mae"], "rmse_entrenamiento": m_train["rmse"],
            "r2_entrenamiento": m_train["r2"],
            "mae_validacion": m_valid["mae"], "rmse_validacion": m_valid["rmse"],
            "r2_validacion": m_valid["r2"],
        })
        modelos[nombre] = modelo
    return pd.DataFrame(filas), modelos


def elegir_mejor(tabla: pd.DataFrame) -> str:
    """Configuracion con menor RMSE de validacion."""
    return str(tabla.loc[tabla["rmse_validacion"].idxmin(), "configuracion"])


def importancias(modelo) -> pd.DataFrame:
    """Importancia de cada posicion del vector de features (suma 1), de mayor a menor."""
    bosque = modelo.stages[-1]
    tabla = pd.DataFrame({"variable": modelado.nombres_features(modelo),
                          "importancia": bosque.featureImportances.toArray()})
    return tabla.sort_values("importancia", ascending=False, ignore_index=True)


def importancias_por_predictor(tabla_imp: pd.DataFrame) -> pd.DataFrame:
    """Suma las importancias de las columnas one-hot para obtener una por cada uno de los seis predictores."""
    predictor = tabla_imp["variable"].str.split("=").str[0]
    return (tabla_imp.groupby(predictor)["importancia"].sum()
            .rename_axis("predictor").sort_values(ascending=False).reset_index())


def cortes_tramos(df: DataFrame, percentiles: list = None) -> list:
    """Percentiles del salario real de `df` que delimitan los tramos."""
    return df.approxQuantile(modelado.OBJETIVO, percentiles or PERCENTILES_TRAMOS, 0.001)


def error_por_tramo(pred: DataFrame, cortes: list) -> pd.DataFrame:
    """n, salario real medio, prediccion media, MAE y error medio (residuo) por tramo de salario real.

    Se calcula con todos los registros de `pred`; el residuo es real - predicho.
    """
    limites = [float("-inf")] + list(cortes) + [float("inf")]
    tramo = F.lit(len(limites) - 2)
    for i in range(len(limites) - 2, 0, -1):
        tramo = F.when(F.col(modelado.OBJETIVO) <= limites[i], F.lit(i - 1)).otherwise(tramo)
    agregado = (modelado.agregar_residuo(pred).withColumn("tramo", tramo)
                .groupBy("tramo")
                .agg(F.count("*").alias("n"),
                     F.avg(modelado.OBJETIVO).alias("salario_real_medio"),
                     F.avg(modelado.PREDICCION).alias("prediccion_media"),
                     F.avg(F.abs(modelado.RESIDUO)).alias("mae"),
                     F.avg(modelado.RESIDUO).alias("error_medio"))
                .toPandas().sort_values("tramo", ignore_index=True))
    agregado.insert(1, "rango", [_rango(limites[t], limites[t + 1]) for t in agregado["tramo"]])
    return agregado


def _rango(desde: float, hasta: float) -> str:
    if desde == float("-inf"):
        return f"<= Q{hasta:,.0f}"
    if hasta == float("inf"):
        return f"> Q{desde:,.0f}"
    return f"Q{desde:,.0f} - Q{hasta:,.0f}"


def guardar_resultados(tabla: pd.DataFrame, mejor: str, modelo, tabla_imp: pd.DataFrame,
                       comparacion: pd.DataFrame, tramos: pd.DataFrame) -> None:
    """Guarda el mejor pipeline y las tablas de la actividad 6."""
    modelado.guardar_modelo(modelo, "random_forest")
    tabla.assign(seleccionada=tabla["configuracion"] == mejor).to_csv(
        config.TABLES / "06_metricas_random_forest.csv", index=False)
    tabla_imp.to_csv(config.TABLES / "06_importancias_random_forest.csv", index=False)
    comparacion.to_csv(config.TABLES / "06_comparacion_validacion.csv", index=False)
    tramos.to_csv(config.TABLES / "06_error_por_tramo_validacion.csv", index=False)
