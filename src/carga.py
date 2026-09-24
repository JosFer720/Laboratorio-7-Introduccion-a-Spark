"""Carga de los xlsx de la ENEIC, homologacion de tipos y union por nombre."""
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src import config

DIR_TIPADO = config.DATA_PROCESSED / "tipado"

_NUMERICAS_DOUBLE = ["salario_mensual", "edad", "antiguedad_anios", "antiguedad_meses",
                     "horas_semanales", "FACTOR"]
_ENTERAS = ["ocupado", "ANIO", "TRIMESTRE"]
_CODIGOS = ["NUM_HOGAR", "NUM_PERSONA", "nivel_educativo", "categoria_ocupacional", "dominio"]


def ruta_archivo(periodo: str, directorio: Path = None) -> Path:
    """Ruta del xlsx de un periodo; error claro si no esta en data/raw."""
    directorio = Path(directorio or config.DATA_RAW)
    ruta = directorio / config.ARCHIVOS[periodo]
    if not ruta.exists():
        disponibles = sorted(p.name for p in directorio.glob("*.xls*"))
        raise FileNotFoundError(f"No existe {ruta}. Archivos en {directorio}: {disponibles}")
    return ruta


def _normalizar(nombre) -> str:
    return str(nombre).strip().upper()


def columnas_del_archivo(ruta: Path, hoja=0) -> list:
    """Encabezados originales de un xlsx (solo lee la primera fila)."""
    return [_normalizar(c) for c in pd.read_excel(ruta, sheet_name=hoja, nrows=0).columns]


def leer_excel(ruta: Path, hoja=0) -> pd.DataFrame:
    """Lee solo las columnas requeridas, todas como texto y con encabezado en mayusculas."""
    requeridas = set(config.COLUMNAS_ORIGINALES)
    pdf = pd.read_excel(ruta, sheet_name=hoja, dtype=str,
                        usecols=lambda c: _normalizar(c) in requeridas)
    pdf.columns = [_normalizar(c) for c in pdf.columns]
    faltantes = [c for c in config.COLUMNAS_ORIGINALES if c not in pdf.columns]
    if faltantes:
        raise ValueError(f"{Path(ruta).name} no trae las columnas: {faltantes}")
    return pdf[config.COLUMNAS_ORIGINALES]


def a_spark_texto(spark: SparkSession, pdf: pd.DataFrame, directorio_tmp: Path = None) -> DataFrame:
    """Pasa un DataFrame de pandas a Spark con esquema explicito de texto (nulos reales).

    Se escribe a un Parquet temporal y Spark lo lee de ahi; asi los datos no pasan por
    workers de Python y el consumo de memoria se mantiene bajo.
    """
    directorio_tmp = Path(directorio_tmp or DIR_TIPADO / "_tmp")
    directorio_tmp.mkdir(parents=True, exist_ok=True)
    esquema = pa.schema([(c, pa.string()) for c in pdf.columns])
    limpio = pdf.astype(object).where(pdf.notna(), None)
    tabla = pa.Table.from_pandas(limpio, schema=esquema, preserve_index=False)
    destino = directorio_tmp / f"{uuid.uuid4().hex}.parquet"
    pq.write_table(tabla, destino)
    return spark.read.parquet(str(destino))


def codigo_canonico(col):
    """Representacion unica de un codigo: '1', '1.0' y ' 1 ' pasan a '1'; vacio pasa a nulo."""
    txt = F.trim(col.cast("string"))
    num = txt.cast("double")
    return (F.when(txt.isNull() | (txt == ""), F.lit(None).cast("string"))
             .when(num.isNotNull() & ~F.isnan(num) & (num == F.floor(num)),
                   num.cast("long").cast("string"))
             .otherwise(txt))


def homologar_tipos(df: DataFrame) -> DataFrame:
    """Renombra a nombres analiticos y fija los tipos de cada columna."""
    for original, nuevo in config.RENOMBRES.items():
        df = df.withColumnRenamed(original, nuevo)
    for c in _NUMERICAS_DOUBLE:
        df = df.withColumn(c, F.trim(F.col(c)).cast("double"))
    for c in _ENTERAS:
        df = df.withColumn(c, F.trim(F.col(c)).cast("double").cast("int"))
    for c in _CODIGOS:
        df = df.withColumn(c, codigo_canonico(F.col(c)))
    return df


def agregar_procedencia(df: DataFrame, periodo: str, archivo: str) -> DataFrame:
    """Identifica el corte publicado al que pertenece el archivo (no corrige TRIMESTRE)."""
    anio, trimestre = config.PERIODOS[periodo]
    return (df.withColumn("periodo_archivo", F.lit(periodo))
              .withColumn("anio_archivo", F.lit(anio).cast("int"))
              .withColumn("trimestre_calendario", F.lit(trimestre).cast("int"))
              .withColumn("archivo_origen", F.lit(archivo)))


def cargar_archivo(spark: SparkSession, periodo: str, ruta: Path = None, hoja=0,
                   directorio_tmp: Path = None) -> DataFrame:
    """Un xlsx -> DataFrame de Spark tipado, con columnas de procedencia."""
    ruta = Path(ruta) if ruta else ruta_archivo(periodo)
    df = homologar_tipos(a_spark_texto(spark, leer_excel(ruta, hoja), directorio_tmp))
    return agregar_procedencia(df, periodo, ruta.name)


def ruta_tipado(periodo: str) -> Path:
    return DIR_TIPADO / f"{periodo}.parquet"


def guardar_tipado(df: DataFrame, periodo: str) -> Path:
    destino = ruta_tipado(periodo)
    df.write.mode("overwrite").parquet(str(destino))
    return destino


def cargar_periodos(spark: SparkSession, periodos: list, reutilizar: bool = True) -> dict:
    """Convierte cada xlsx una sola vez a Parquet tipado y devuelve {periodo: DataFrame}.

    Los archivos se procesan de uno en uno para controlar la memoria.
    """
    dfs = {}
    for periodo in periodos:
        destino = ruta_tipado(periodo)
        if not (reutilizar and (destino / "_SUCCESS").exists()):
            guardar_tipado(cargar_archivo(spark, periodo), periodo)
            shutil.rmtree(DIR_TIPADO / "_tmp", ignore_errors=True)
        dfs[periodo] = spark.read.parquet(str(destino))
    return dfs


def unir_periodos(dfs: list) -> DataFrame:
    """Apila por nombre de columna (nunca por posicion)."""
    resultado = dfs[0]
    for df in dfs[1:]:
        resultado = resultado.unionByName(df)
    return resultado
