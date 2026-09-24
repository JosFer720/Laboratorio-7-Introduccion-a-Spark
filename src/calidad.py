"""Diagnostico de calidad, filtros en orden fijo y verificacion de unicidad."""
import pandas as pd
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from src import config


def es_finito(nombre: str) -> Column:
    """Verdadero solo si la columna numerica no es nula, NaN ni infinita."""
    c = F.col(nombre)
    return c.isNotNull() & ~F.isnan(c) & (c != F.lit(float("inf"))) & (c != F.lit(float("-inf")))


def agregar_antiguedad(df: DataFrame) -> DataFrame:
    """Antiguedad en anios = anios + meses / 12 (nula si falta alguno de los dos)."""
    return df.withColumn("antiguedad", F.col("antiguedad_anios") + F.col("antiguedad_meses") / 12)


def faltantes(df: DataFrame, columnas: list) -> pd.DataFrame:
    """Cantidad y porcentaje de faltantes (nulo o NaN) por columna, sin filtrar."""
    tipos = dict(df.dtypes)
    exprs = [F.count(F.lit(1)).alias("__total")]
    for c in columnas:
        vacio = F.col(c).isNull()
        if tipos[c] in ("double", "float"):
            vacio = vacio | F.isnan(F.col(c))
        exprs.append(F.sum(vacio.cast("int")).alias(c))
    fila = df.agg(*exprs).first()
    total = fila["__total"]
    return pd.DataFrame({
        "variable": columnas,
        "faltantes": [int(fila[c] or 0) for c in columnas],
        "porcentaje": [round(100 * (fila[c] or 0) / total, 2) if total else 0.0 for c in columnas],
        "registros": total,
    })


def pasos_filtro() -> list:
    """Filtros de la poblacion analitica, siempre en este orden. Requiere `antiguedad`."""
    meses = F.col("antiguedad_meses")
    pasos = [
        ("edad finita y >= 15",
         es_finito("edad") & (F.col("edad") >= config.EDAD_MINIMA)),
        ("ocupado (OCUPADOS = 1)",
         F.col("ocupado") == 1),
        ("asalariado (P05C16 en 1-4)",
         F.col("categoria_ocupacional").isin([str(c) for c in config.CATEGORIAS_ASALARIADO])),
        ("salario finito y > 0",
         es_finito("salario_mensual") & (F.col("salario_mensual") > 0)),
        ("antiguedad valida (meses 0-11, >= 0 y <= edad)",
         es_finito("antiguedad_anios") & es_finito("antiguedad_meses")
         & (meses == F.floor(meses)) & (meses >= 0) & (meses <= 11)
         & (F.col("antiguedad_anios") >= 0)
         & (F.col("antiguedad") >= 0) & (F.col("antiguedad") <= F.col("edad"))),
        ("horas habituales > 0 y <= 168",
         es_finito("horas_semanales") & (F.col("horas_semanales") > 0)
         & (F.col("horas_semanales") <= config.HORAS_MAXIMAS)),
    ]
    # Lo que no permite evaluar el criterio (nulo) se excluye y se contabiliza.
    return [(nombre, F.coalesce(cond, F.lit(False))) for nombre, cond in pasos]


def embudo(df: DataFrame) -> pd.DataFrame:
    """Registros que sobreviven y se excluyen en cada paso, por periodo y en total."""
    df = agregar_antiguedad(df)
    pasos = pasos_filtro()
    acumulado = F.lit(True)
    exprs = [F.count(F.lit(1)).alias("inicial")]
    for i, (_, cond) in enumerate(pasos, start=1):
        acumulado = acumulado & cond
        exprs.append(F.sum(acumulado.cast("int")).alias(f"paso_{i}"))
    ancho = df.groupBy("periodo_archivo").agg(*exprs).orderBy("periodo_archivo").toPandas()
    columnas = ["inicial"] + [f"paso_{i}" for i in range(1, len(pasos) + 1)]
    total = ancho[columnas].sum()
    total["periodo_archivo"] = "TOTAL"
    ancho = pd.concat([ancho, total.to_frame().T], ignore_index=True)

    filas = []
    for _, fila in ancho.iterrows():
        previo = int(fila["inicial"])
        filas.append((fila["periodo_archivo"], "registros originales", previo, 0))
        for i, (nombre, _) in enumerate(pasos, start=1):
            restante = int(fila[f"paso_{i}"] or 0)
            filas.append((fila["periodo_archivo"], nombre, restante, previo - restante))
            previo = restante
    return pd.DataFrame(filas, columns=["periodo_archivo", "paso", "restantes", "excluidos"])


def resumen_por_archivo(tabla_embudo: pd.DataFrame) -> pd.DataFrame:
    """Registros por archivo antes y despues de todos los filtros."""
    antes = tabla_embudo[tabla_embudo["paso"] == "registros originales"].set_index("periodo_archivo")["restantes"]
    despues = tabla_embudo.groupby("periodo_archivo")["restantes"].min()
    resultado = pd.DataFrame({"antes": antes, "despues": despues.reindex(antes.index)})
    resultado["retenido_pct"] = (100 * resultado["despues"] / resultado["antes"]).round(2)
    return resultado.reset_index()


def normalizar_categoricas(df: DataFrame) -> DataFrame:
    """Valida los codigos contra el diccionario; ausentes o no reconocidos -> DESCONOCIDO."""
    for c, validos in config.CODIGOS_VALIDOS.items():
        df = df.withColumn(
            c, F.when(F.col(c).isin(validos), F.col(c)).otherwise(F.lit(config.DESCONOCIDO)))
    return df


def codigos_observados(df: DataFrame, columna: str) -> pd.DataFrame:
    """Frecuencia de cada codigo observado y si el diccionario lo reconoce."""
    validos = config.CODIGOS_VALIDOS[columna]
    tabla = (df.groupBy(F.coalesce(F.col(columna), F.lit("<AUSENTE>")).alias("codigo"))
               .count().orderBy("codigo").toPandas())
    tabla["reconocido"] = tabla["codigo"].isin(validos)
    return tabla


def preparar(df: DataFrame) -> DataFrame:
    """Poblacion analitica: filtros en orden fijo, categoricas validadas y columnas finales."""
    df = agregar_antiguedad(df)
    for _, cond in pasos_filtro():
        df = df.filter(cond)
    df = normalizar_categoricas(df)
    return df.select(*config.COLUMNAS_PREPARADAS)


def resumen_unicidad(df: DataFrame) -> dict:
    """Verifica la unicidad de (periodo_archivo, NUM_HOGAR, NUM_PERSONA)."""
    por_clave = df.groupBy(*config.CLAVE).count()
    fila = por_clave.agg(
        F.count(F.lit(1)).alias("claves"),
        F.sum((F.col("count") > 1).cast("int")).alias("claves_duplicadas"),
        F.sum(F.when(F.col("count") > 1, F.col("count")).otherwise(0)).alias("filas_en_duplicadas"),
    ).first()
    return {
        "registros": df.count(),
        "claves_distintas": fila["claves"],
        "claves_duplicadas": int(fila["claves_duplicadas"] or 0),
        "filas_en_claves_duplicadas": int(fila["filas_en_duplicadas"] or 0),
    }


def detalle_duplicados(df: DataFrame) -> DataFrame:
    """Por cada clave repetida: cuantas filas hay y si son repeticion exacta o conflicto.

    Exacta: todas las filas de la clave son identicas en las columnas seleccionadas.
    Conflicto: la misma clave trae valores distintos. No se elimina nada.
    """
    datos = [c for c in df.columns if c not in config.CLAVE]
    version = F.to_json(F.struct(*[F.col(c) for c in datos]))
    return (df.groupBy(*config.CLAVE)
              .agg(F.count(F.lit(1)).alias("filas"),
                   F.size(F.collect_set(version)).alias("versiones"))
              .filter(F.col("filas") > 1)
              .withColumn("tipo", F.when(F.col("versiones") == 1, "repeticion exacta")
                                   .otherwise("conflicto")))


def columnas_en_conflicto(df: DataFrame) -> pd.DataFrame:
    """En cuantas claves repetidas difiere cada columna."""
    datos = [c for c in df.columns if c not in config.CLAVE]
    exprs = [(F.size(F.collect_set(F.coalesce(F.col(c).cast("string"), F.lit("<NULO>")))) > 1).alias(c)
             for c in datos]
    por_clave = df.groupBy(*config.CLAVE).agg(F.count(F.lit(1)).alias("filas"), *exprs)
    por_clave = por_clave.filter(F.col("filas") > 1)
    fila = por_clave.agg(*[F.sum(F.col(c).cast("int")).alias(c) for c in datos]).first()
    return (pd.DataFrame({"columna": datos, "claves_con_diferencia": [int(fila[c] or 0) for c in datos]})
              .sort_values("claves_con_diferencia", ascending=False, ignore_index=True))


def guardar_preparado(df: DataFrame, nombre: str):
    """Guarda el conjunto preparado (personas_2025 / personas_2026) en Parquet."""
    destino = config.DATA_PROCESSED / f"{nombre}.parquet"
    df.write.mode("overwrite").parquet(str(destino))
    return destino
