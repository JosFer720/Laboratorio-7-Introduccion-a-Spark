"""Entrenamiento final con 2025, evaluacion en 2026 (act. 7) y analisis de errores (act. 8)."""
import pandas as pd
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src import config, exploracion, modelado, random_forest, regresion_lineal

MODELOS = ["regresion_lineal", "random_forest"]
NOMBRES = {"referencia": "referencia", "regresion_lineal": "regresión lineal",
           "random_forest": "random forest"}
PREDICCIONES = {m: f"pred_{m}" for m in ["referencia"] + MODELOS}
RESIDUOS = {m: f"residuo_{m}" for m in MODELOS}
N_MUESTRA = 5000
# Percentiles para describir la distribucion del salario y cortes de los tramos de error en prueba.
PERCENTILES_SALARIO = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
PERCENTILES_TRAMOS = [0.25, 0.50, 0.75, 0.90, 0.95]

# Etiquetas del diccionario de Personas de la ENEIC.
ETIQUETAS = {
    "nivel_educativo": {
        "0": "ninguno", "1": "preprimaria", "2": "primaria", "3": "básico", "4": "diversificado",
        "5": "superior", "6": "maestría", "7": "doctorado",
    },
    "categoria_ocupacional": {
        "1": "empleado de gobierno", "2": "empleado de empresa privada",
        "3": "jornalero o peón", "4": "servicio doméstico",
    },
    "dominio": {"1": "urbano metropolitano", "2": "resto urbano", "3": "rural nacional"},
}


def parametros_lineal(tabla: pd.DataFrame = None) -> tuple:
    """(configuracion, regParam, elasticNetParam) con menor RMSE de validacion de la actividad 5."""
    if tabla is None:
        tabla = pd.read_csv(config.TABLES / "05_metricas_regresion_lineal.csv")
    fila = tabla.set_index("configuracion").loc[regresion_lineal.elegir_mejor(tabla)]
    return fila.name, float(fila["regParam"]), float(fila["elasticNetParam"])


def parametros_random_forest(tabla: pd.DataFrame = None) -> tuple:
    """(configuracion, numTrees, maxDepth) con menor RMSE de validacion de la actividad 6."""
    if tabla is None:
        tabla = pd.read_csv(config.TABLES / "06_metricas_random_forest.csv")
    fila = tabla.set_index("configuracion").loc[random_forest.elegir_mejor(tabla)]
    return fila.name, int(fila["numTrees"]), int(fila["maxDepth"])


def entrenar_finales(desarrollo: DataFrame, lineal: tuple, bosque: tuple) -> dict:
    """Reajusta cada configuracion elegida con los cuatro trimestres de 2025.

    `lineal` = (regParam, elasticNetParam) y `bosque` = (numTrees, maxDepth). Falla si el conjunto
    trae periodos que no son de 2025, para que la prueba no entre al entrenamiento.
    """
    periodos = {r[0] for r in desarrollo.select("periodo_archivo").distinct().collect()}
    fuera = periodos - set(config.PERIODOS_TRAIN)
    if fuera:
        raise ValueError(f"El entrenamiento final solo puede usar 2025; sobran: {sorted(fuera)}")
    return {
        "regresion_lineal": regresion_lineal.pipeline_lineal(*lineal).fit(desarrollo),
        "random_forest": random_forest.pipeline_random_forest(*bosque).fit(desarrollo),
    }


def predecir(modelos: dict, prueba: DataFrame, valor_referencia: float) -> DataFrame:
    """Predicciones de la referencia y de cada modelo sobre los mismos registros de `prueba`.

    Los modelos se aplican uno tras otro sobre el mismo DataFrame, sin uniones, de modo que todas las
    predicciones corresponden a exactamente las mismas filas. Agrega el residuo (real - predicho).
    """
    pred = prueba.withColumn(PREDICCIONES["referencia"], F.lit(float(valor_referencia)))
    for nombre in MODELOS:
        anteriores = pred.columns
        pred = modelos[nombre].transform(pred).select(
            *anteriores, F.col(modelado.PREDICCION).alias(PREDICCIONES[nombre]))
    for nombre in MODELOS:
        pred = pred.withColumn(RESIDUOS[nombre], F.col(modelado.OBJETIVO) - F.col(PREDICCIONES[nombre]))
    return pred


def metricas_prueba(pred: DataFrame) -> pd.DataFrame:
    """MAE, RMSE, R2 y n de la referencia y de cada modelo, sobre todos los registros de `pred`."""
    return modelado.tabla_metricas({
        NOMBRES[m]: modelado.metricas(pred, prediccion=PREDICCIONES[m]) for m in ["referencia"] + MODELOS})


def muestra_comun(pred: DataFrame, n: int = N_MUESTRA, semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Una sola muestra de hasta `n` registros con las predicciones de ambos modelos (solo para graficar)."""
    columnas = ([modelado.OBJETIVO] + config.CATEGORICAS + [PREDICCIONES[m] for m in MODELOS]
                + [RESIDUOS[m] for m in MODELOS])
    return exploracion.muestra(pred.select(*columnas), n=n, semilla=semilla)


def error_por_grupo(pred: DataFrame, columna: str) -> pd.DataFrame:
    """n, salario real medio y, por modelo, prediccion media, MAE y error medio por categoria de `columna`.

    Se calcula con todos los registros de `pred`. Error medio = promedio de (real - predicho).
    """
    agregados = [F.count("*").alias("n"), F.avg(modelado.OBJETIVO).alias("salario_real_medio")]
    for m in MODELOS:
        agregados += [F.avg(PREDICCIONES[m]).alias(f"prediccion_media_{m}"),
                      F.avg(F.abs(RESIDUOS[m])).alias(f"mae_{m}"),
                      F.avg(RESIDUOS[m]).alias(f"error_medio_{m}")]
    tabla = pred.groupBy(columna).agg(*agregados).toPandas()
    tabla = tabla.rename(columns={columna: "codigo"})
    tabla.insert(1, "categoria", tabla["codigo"].map(ETIQUETAS.get(columna, {})).fillna(tabla["codigo"]))
    tabla.insert(0, "variable", columna)
    orden = pd.to_numeric(tabla["codigo"], errors="coerce").fillna(float("inf"))
    return tabla.assign(_orden=orden).sort_values("_orden", ignore_index=True).drop(columns="_orden")


def percentiles_salario(conjuntos: dict, percentiles: list = None) -> pd.DataFrame:
    """Media, desviacion y percentiles del salario real de cada conjunto ({nombre: DataFrame})."""
    percentiles = percentiles or PERCENTILES_SALARIO
    columnas = {}
    for nombre, df in conjuntos.items():
        fila = df.agg(F.count("*").alias("n"), F.avg(modelado.OBJETIVO).alias("media"),
                      F.stddev(modelado.OBJETIVO).alias("desv_estandar"),
                      F.max(modelado.OBJETIVO).alias("maximo")).first().asDict()
        valores = df.approxQuantile(modelado.OBJETIVO, percentiles, 0.001)
        fila.update({f"p{round(100 * p)}": v for p, v in zip(percentiles, valores)})
        columnas[nombre] = fila
    tabla = pd.DataFrame(columnas)
    tabla.index.name = "estadistico"
    return tabla


def rango_predicciones(pred: DataFrame) -> pd.DataFrame:
    """Minimo, percentiles 1 y 99 y maximo del salario real y de lo predicho por cada modelo."""
    columnas = {"salario real": modelado.OBJETIVO}
    columnas.update({NOMBRES[m]: PREDICCIONES[m] for m in MODELOS})
    filas = {}
    for nombre, col in columnas.items():
        minimo, maximo = pred.agg(F.min(col), F.max(col)).first()
        p1, p99 = pred.approxQuantile(col, [0.01, 0.99], 0.001)
        filas[nombre] = {"minimo": minimo, "p1": p1, "p99": p99, "maximo": maximo,
                         "negativos": pred.filter(F.col(col) < 0).count()}
    return pd.DataFrame(filas).T


def error_por_decil_prediccion(pred: DataFrame, modelo: str, grupos: int = 10) -> pd.DataFrame:
    """n, prediccion media, salario real medio, MAE y error medio por decil de lo predicho por `modelo`."""
    columna = PREDICCIONES[modelo]
    cortes = sorted(set(pred.approxQuantile(columna, [i / grupos for i in range(1, grupos)], 0.001)))
    decil = Bucketizer(splits=[float("-inf")] + cortes + [float("inf")], inputCol=columna, outputCol="decil")
    return (decil.transform(pred).groupBy("decil")
            .agg(F.count("*").alias("n"), F.avg(columna).alias("prediccion_media"),
                 F.avg(modelado.OBJETIVO).alias("salario_real_medio"),
                 F.avg(F.abs(RESIDUOS[modelo])).alias("mae"), F.avg(RESIDUOS[modelo]).alias("error_medio"))
            .orderBy("decil").toPandas())


def resumen_errores(pred: DataFrame, umbral_alto: float, umbral_cola: float,
                    tolerancia: float = 0.2) -> pd.DataFrame:
    """Tamano y signo de los errores de cada modelo, con todos los registros de `pred`.

    - mediana del error absoluto y % de registros con error menor o igual a `tolerancia` del salario real;
    - % subestimados (residuo > 0) en total y entre los salarios reales mayores que `umbral_alto`;
    - % de la suma de errores cuadraticos que aportan los salarios reales mayores que `umbral_cola`.
    """
    real = F.col(modelado.OBJETIVO)
    filas = {}
    for m in MODELOS:
        residuo = F.col(RESIDUOS[m])
        fila = pred.agg(
            F.avg((F.abs(residuo) <= tolerancia * real).cast("double")).alias("dentro"),
            F.avg((residuo > 0).cast("double")).alias("sub_total"),
            F.avg(F.when(real > umbral_alto, (residuo > 0).cast("double"))).alias("sub_alto"),
            F.sum(residuo * residuo).alias("sse"),
            F.sum(F.when(real > umbral_cola, residuo * residuo)).alias("sse_cola"),
        ).first()
        mediana = pred.select(F.abs(residuo).alias("a")).approxQuantile("a", [0.5], 0.001)[0]
        filas[NOMBRES[m]] = {
            "mediana_error_absoluto": mediana,
            f"pct_error_hasta_{round(100 * tolerancia)}pct": 100 * fila["dentro"],
            "pct_subestimados": 100 * fila["sub_total"],
            "pct_subestimados_salario_alto": 100 * fila["sub_alto"],
            "pct_error_cuadratico_cola": 100 * (fila["sse_cola"] or 0.0) / fila["sse"],
        }
    tabla = pd.DataFrame(filas).T
    tabla.index.name = "modelo"
    return tabla.reset_index()


def error_por_tramo(pred: DataFrame, cortes: list) -> pd.DataFrame:
    """MAE y error medio por tramo de salario real para cada modelo (reutiliza la funcion de la act. 6)."""
    tablas = [random_forest.error_por_tramo(pred.withColumn(modelado.PREDICCION, F.col(PREDICCIONES[m])), cortes)
              .assign(modelo=NOMBRES[m]) for m in MODELOS]
    return pd.concat(tablas, ignore_index=True)


def cortes_tramos(pred: DataFrame, percentiles: list = None) -> list:
    """Percentiles del salario real de prueba que delimitan los tramos."""
    return random_forest.cortes_tramos(pred, percentiles or PERCENTILES_TRAMOS)


def guardar_modelos(modelos: dict) -> list:
    """Guarda los pipelines finales en results/modelos/<modelo>_final."""
    return [modelado.guardar_modelo(modelo, f"{nombre}_final") for nombre, modelo in modelos.items()]
