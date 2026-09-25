"""Pipeline de regresion lineal regularizada para estimar salario_mensual."""
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame

from src import config, modelado

MAX_ITER = 100
# nombre -> (regParam, elasticNetParam). elasticNetParam 0 = Ridge (L2), 1 = Lasso (L1).
CONFIGURACIONES = {
    "sin_regularizacion": (0.0, 0.0),
    "ridge_0.01": (0.01, 0.0),
    "ridge_0.1": (0.1, 0.0),
    "lasso_0.01": (0.01, 1.0),
    "elasticnet_0.01_0.5": (0.01, 0.5),
}


def pipeline_lineal(reg_param: float, elastic_net: float, max_iter: int = MAX_ITER) -> Pipeline:
    """Preprocesamiento categorico + LinearRegression con estandarizacion interna de los predictores."""
    regresion = LinearRegression(
        featuresCol="features", labelCol=modelado.OBJETIVO, predictionCol=modelado.PREDICCION,
        regParam=reg_param, elasticNetParam=elastic_net, standardization=True,
        fitIntercept=True, maxIter=max_iter)
    return Pipeline(stages=modelado.etapas_preprocesamiento() + [regresion])


def evaluar_configuraciones(entrenamiento: DataFrame, validacion: DataFrame,
                            configuraciones: dict = None) -> tuple:
    """Ajusta cada configuracion solo con entrenamiento y la evalua en entrenamiento y validacion.

    Devuelve (tabla de metricas, {nombre: PipelineModel}).
    """
    configuraciones = configuraciones or CONFIGURACIONES
    filas, modelos = [], {}
    for nombre, (reg_param, elastic_net) in configuraciones.items():
        modelo = pipeline_lineal(reg_param, elastic_net).fit(entrenamiento)
        m_train = modelado.metricas(modelo.transform(entrenamiento))
        m_valid = modelado.metricas(modelo.transform(validacion))
        filas.append({
            "configuracion": nombre, "regParam": reg_param, "elasticNetParam": elastic_net,
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


def coeficientes(modelo) -> pd.DataFrame:
    """Intercepto y coeficientes (en las unidades originales de cada predictor)."""
    regresion = modelo.stages[-1]
    nombres = modelado.nombres_features(modelo)
    tabla = pd.DataFrame({"variable": nombres, "coeficiente": list(regresion.coefficients)})
    tabla = tabla.sort_values("coeficiente", key=abs, ascending=False, ignore_index=True)
    intercepto = pd.DataFrame({"variable": ["(intercepto)"], "coeficiente": [regresion.intercept]})
    return pd.concat([intercepto, tabla], ignore_index=True)


def guardar_resultados(tabla: pd.DataFrame, mejor: str, modelo, tabla_coef: pd.DataFrame) -> None:
    """Guarda el mejor pipeline y las tablas de la actividad 5."""
    modelado.guardar_modelo(modelo, "regresion_lineal")
    tabla.assign(seleccionada=tabla["configuracion"] == mejor).to_csv(
        config.TABLES / "05_metricas_regresion_lineal.csv", index=False)
    tabla_coef.to_csv(config.TABLES / "05_coeficientes_regresion_lineal.csv", index=False)
