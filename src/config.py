"""Configuracion compartida: rutas, semilla, sesion de Spark y catalogos de columnas."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
RESULTS = RAIZ / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
MODELOS = RESULTS / "modelos"

SEMILLA = 42
SPARK_DRIVER_MEMORY = "4g"
SHUFFLE_PARTITIONS = 16

# periodo_archivo -> (anio, trimestre calendario)
PERIODOS = {
    "2025T1": (2025, 1),
    "2025T2": (2025, 2),
    "2025T3": (2025, 3),
    "2025T4": (2025, 4),
    "2026T1": (2026, 1),
}
# Nombre del xlsx de Personas de cada periodo dentro de data/raw
ARCHIVOS = {
    "2025T1": "ENEIC_2025T1.xlsx",
    "2025T2": "ENEIC_2025T2.xlsx",
    "2025T3": "ENEIC_2025T3.xlsx",
    "2025T4": "ENEIC_2025T4.xlsx",
    "2026T1": "ENEIC_2026T1.xlsx",
}
PERIODOS_TRAIN =["2025T1", "2025T2", "2025T3", "2025T4"]
PERIODO_TEST = "2026T1"

# Columnas originales que se seleccionan de cada archivo
COLUMNAS_ORIGINALES = [
    "P05D01", "P02A03", "P05C07A", "P05C07B", "P05H01A", "P03A03A",
    "P05C16", "DOMINIO", "OCUPADOS", "NUM_HOGAR", "NUM_PERSONA",
    "FACTOR", "ANIO", "TRIMESTRE",
]

# Columnas originales -> nombre analitico
RENOMBRES = {
    "P05D01": "salario_mensual",
    "P02A03": "edad",
    "P05C07A": "antiguedad_anios",
    "P05C07B": "antiguedad_meses",
    "P05H01A": "horas_semanales",
    "P03A03A": "nivel_educativo",
    "P05C16": "categoria_ocupacional",
    "DOMINIO": "dominio",
    "OCUPADOS": "ocupado",
}

CATEGORICAS = ["nivel_educativo", "categoria_ocupacional", "dominio"]
NUMERICAS = ["edad", "antiguedad", "horas_semanales"]
CATEGORIAS_ASALARIADO = [1, 2, 3, 4]

# Codigos validos segun el diccionario de datos (el codigo educativo 0 es "ninguno").
CODIGOS_VALIDOS = {
    "nivel_educativo": [str(i) for i in range(0, 10)],
    "categoria_ocupacional": [str(c) for c in CATEGORIAS_ASALARIADO],
    "dominio": [str(i) for i in range(1, 4)],
}
DESCONOCIDO = "DESCONOCIDO"

EDAD_MINIMA = 15
HORAS_MAXIMAS = 168

COLUMNAS_PREPARADAS = [
    "periodo_archivo", "anio_archivo", "trimestre_calendario", "archivo_origen",
    "NUM_HOGAR", "NUM_PERSONA", "FACTOR", "ANIO", "TRIMESTRE",
    "salario_mensual", "edad", "antiguedad", "horas_semanales",
    "nivel_educativo", "categoria_ocupacional", "dominio",
]
CLAVE = ["periodo_archivo", "NUM_HOGAR", "NUM_PERSONA"]


def crear_spark(nombre: str = "lab7"):
    """Crea (o reutiliza) la sesion de Spark local."""
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[*]")
        .appName(nombre)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
