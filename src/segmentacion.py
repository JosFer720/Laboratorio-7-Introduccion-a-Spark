"""Segmentacion de perfiles con KMeans."""
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src import config
from src.exploracion import muestra

VARIABLES_PERFIL = ["edad", "antiguedad", "horas_semanales"]
# El salario entra en log10 para que la cola derecha no domine las distancias.
SALARIO_LOG = "salario_log10"
VARIABLES_CON_SALARIO = VARIABLES_PERFIL + [SALARIO_LOG]
ESPECIFICACIONES = {
    "sin_salario": VARIABLES_PERFIL,
    "con_salario": VARIABLES_CON_SALARIO,
}
KS = [2, 3, 4, 5]
TAMANO_MINIMO_PCT = 5.0
MAX_ITER = 50

ETIQUETAS = {
    "edad": "Edad (años)",
    "antiguedad": "Antigüedad (años)",
    "horas_semanales": "Horas semanales",
    SALARIO_LOG: "log10(salario)",
    "salario_mensual": "Salario mensual (Q)",
}


def agregar_salario_log(df: DataFrame) -> DataFrame:
    """log10 del salario, solo como variable de segmentacion (el objetivo no se transforma)."""
    return df.withColumn(SALARIO_LOG, F.log10("salario_mensual"))


def pipeline_kmeans(variables: list, k: int, semilla: int = config.SEMILLA) -> Pipeline:
    """VectorAssembler -> StandardScaler (media 0, desv. 1) -> KMeans."""
    return Pipeline(stages=[
        VectorAssembler(inputCols=variables, outputCol="x", handleInvalid="error"),
        StandardScaler(inputCol="x", outputCol="x_std", withMean=True, withStd=True),
        KMeans(featuresCol="x_std", predictionCol="cluster", k=k, seed=semilla,
               maxIter=MAX_ITER, initMode="k-means||"),
    ])


def evaluar_k(df: DataFrame, variables: list, ks: list = None,
              semilla: int = config.SEMILLA) -> tuple:
    """Ajusta KMeans para cada k y calcula silhouette, WSSSE y tamanos sobre todos los registros.

    Devuelve (tabla de metricas, {k: PipelineModel}).
    """
    ks = ks or KS
    evaluador = ClusteringEvaluator(featuresCol="x_std", predictionCol="cluster",
                                    metricName="silhouette", distanceMeasure="squaredEuclidean")
    filas, modelos = [], {}
    for k in ks:
        modelo = pipeline_kmeans(variables, k, semilla).fit(df)
        resumen = modelo.stages[-1].summary
        tamanos = resumen.clusterSizes
        total = sum(tamanos)
        filas.append({
            "k": k,
            "silhouette": evaluador.evaluate(modelo.transform(df)),
            "wssse": resumen.trainingCost,
            "cluster_menor_pct": 100 * min(tamanos) / total,
            "cluster_mayor_pct": 100 * max(tamanos) / total,
            "tamanos": list(tamanos),
        })
        modelos[k] = modelo
    tabla = pd.DataFrame(filas)
    tabla["wssse_reduccion_pct"] = (-100 * tabla["wssse"].pct_change()).round(2)
    return tabla, modelos


def elegir_k(tabla: pd.DataFrame, tamano_minimo_pct: float = TAMANO_MINIMO_PCT,
             k_minimo: int = 3) -> int:
    """Codo del WSSSE: el k (>= `k_minimo`) cuya reduccion del WSSSE frente a k-1 es la mayor.

    Solo se consideran los k cuyo cluster mas pequeno tiene al menos `tamano_minimo_pct` % de
    los registros. K = 2 queda fuera porque solo parte los datos en dos mitades y no permite
    distinguir perfiles. Si ningun k cumple, se usa el de mayor silhouette.
    """
    validos = tabla[(tabla["k"] >= k_minimo) & (tabla["cluster_menor_pct"] >= tamano_minimo_pct)]
    if validos["wssse_reduccion_pct"].notna().any():
        return int(validos.loc[validos["wssse_reduccion_pct"].idxmax(), "k"])
    return int(tabla.loc[tabla["silhouette"].idxmax(), "k"])


def centroides(modelo: PipelineModel, variables: list) -> pd.DataFrame:
    """Centroides en unidades estandarizadas (z) y en unidades originales."""
    escala = modelo.stages[1]
    media, desv = escala.mean.toArray(), escala.std.toArray()
    filas = []
    for i, centro in enumerate(modelo.stages[-1].clusterCenters()):
        fila = {"cluster": i}
        for j, v in enumerate(variables):
            fila[f"{v}_z"] = float(centro[j])
            fila[v] = float(centro[j] * desv[j] + media[j])
        filas.append(fila)
    return pd.DataFrame(filas).set_index("cluster")


def perfil_clusters(pred: DataFrame) -> pd.DataFrame:
    """Tamano, medias y medianas de las variables numericas y P25/P75 del salario por cluster."""
    exprs = [F.count(F.lit(1)).alias("registros")]
    for v in VARIABLES_PERFIL + ["salario_mensual"]:
        exprs += [F.mean(v).alias(f"{v}_media"),
                  F.expr(f"percentile({v}, 0.5)").alias(f"{v}_mediana")]
    exprs += [F.expr("percentile(salario_mensual, 0.25)").alias("salario_mensual_p25"),
              F.expr("percentile(salario_mensual, 0.75)").alias("salario_mensual_p75")]
    tabla = pred.groupBy("cluster").agg(*exprs).orderBy("cluster").toPandas().set_index("cluster")
    tabla.insert(1, "porcentaje", (100 * tabla["registros"] / tabla["registros"].sum()).round(2))
    return tabla


def composicion(pred: DataFrame, columna: str) -> pd.DataFrame:
    """Porcentaje de cada categoria de `columna` dentro de cada cluster (filas suman 100)."""
    conteos = pred.groupBy("cluster", columna).count().toPandas()
    tabla = conteos.pivot(index="cluster", columns=columna, values="count").fillna(0)
    return (100 * tabla.div(tabla.sum(axis=1), axis=0)).round(2)


def predominante(tabla_composicion: pd.DataFrame) -> pd.DataFrame:
    """Categoria mas frecuente de cada cluster y su porcentaje."""
    return pd.DataFrame({"categoria": tabla_composicion.idxmax(axis=1),
                         "porcentaje": tabla_composicion.max(axis=1)})


def _nivel(z: float, bajo: str, alto: str, umbral: float = 0.5) -> str:
    if z >= umbral:
        return alto
    if z <= -umbral:
        return bajo
    return ""


def nombrar_clusters(cent: pd.DataFrame) -> pd.Series:
    """Nombre corto de cada cluster a partir de su centroide estandarizado.

    Un rasgo se menciona cuando el centroide se aleja al menos media desviacion estandar
    del promedio de los registros analizados.
    """
    nombres = {}
    for c, fila in cent.iterrows():
        rasgos = [
            _nivel(fila["edad_z"], "jóvenes", "mayores"),
            _nivel(fila["antiguedad_z"], "poca antigüedad", "antigüedad alta"),
            _nivel(fila["horas_semanales_z"], "jornada corta", "jornada larga"),
        ]
        if f"{SALARIO_LOG}_z" in fila:
            rasgos.append(_nivel(fila[f"{SALARIO_LOG}_z"], "salario bajo", "salario alto"))
        rasgos = [r for r in rasgos if r]
        nombres[c] = ", ".join(rasgos).capitalize() if rasgos else "Perfil promedio"
    return pd.Series(nombres, name="nombre")


def proyeccion_pca(pred: DataFrame, n: int = 5000, semilla: int = config.SEMILLA) -> tuple:
    """PCA de 2 componentes ajustado con todos los registros; devuelve una muestra de hasta `n`.

    Retorna (pandas con pc1, pc2 y cluster, varianza explicada por componente).
    """
    pca = PCA(k=2, inputCol="x_std", outputCol="pc").fit(pred)
    proy = (pca.transform(pred)
               .withColumn("pc_arr", vector_to_array("pc"))
               .select(F.col("pc_arr")[0].alias("pc1"), F.col("pc_arr")[1].alias("pc2"), "cluster",
                       *VARIABLES_PERFIL, "salario_mensual"))
    return muestra(proy, n, semilla), list(pca.explainedVariance.toArray())

