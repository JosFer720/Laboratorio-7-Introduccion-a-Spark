"""Pruebas de src.carga."""
import pandas as pd
import pytest

from src import carga, config


def _xlsx(ruta, filas, extra=None, invertir=False):
    columnas = list(config.COLUMNAS_ORIGINALES)
    pdf = pd.DataFrame(filas, columns=columnas)
    if extra:
        for nombre in extra:
            pdf[nombre] = "x"
    if invertir:
        pdf = pdf[list(reversed(pdf.columns))]
    pdf.to_excel(ruta, index=False)
    return ruta


# P05D01, P02A03, P05C07A, P05C07B, P05H01A, P03A03A, P05C16, DOMINIO, OCUPADOS,
# NUM_HOGAR, NUM_PERSONA, FACTOR, ANIO, TRIMESTRE
FILAS = [
    ["3000", "30", "2", "6", "40", "0", "2.0", "1", "1", "10", "1", "150.5", "2025", "2"],
    [None, "45", None, None, None, "3", "1", "2", "1", "11", "2", "120", "2025", "2"],
]


def test_codigo_canonico_unifica_representaciones(spark):
    df = spark.sql("SELECT * FROM VALUES ('1', 1), ('1.0', 2), (' 1 ', 3), ('', 4), "
                   "(CAST(NULL AS STRING), 5), ('abc', 6) AS t(c, i)")
    salida = [r[0] for r in df.orderBy("i").select(carga.codigo_canonico(df.c)).collect()]
    assert salida == ["1", "1", "1", None, None, "abc"]


def test_cargar_archivo_tipa_y_agrega_procedencia(spark, tmp_path):
    ruta = _xlsx(tmp_path / "a.xlsx", FILAS)
    df = carga.cargar_archivo(spark, "2025T1", ruta, directorio_tmp=tmp_path / "t")
    tipos = dict(df.dtypes)
    assert tipos["salario_mensual"] == "double"
    assert tipos["edad"] == "double"
    assert tipos["ANIO"] == "int"
    assert tipos["categoria_ocupacional"] == "string"
    fila = df.orderBy("NUM_HOGAR").first()
    assert fila["periodo_archivo"] == "2025T1"
    assert fila["anio_archivo"] == 2025 and fila["trimestre_calendario"] == 1
    assert fila["archivo_origen"] == "a.xlsx"
    assert fila["categoria_ocupacional"] == "2"  # '2.0' llega como '2'
    assert fila["nivel_educativo"] == "0"        # el cero se conserva
    assert fila["TRIMESTRE"] == 2                # se conserva el valor original


def test_faltantes_llegan_como_nulos(spark, tmp_path):
    df = carga.cargar_archivo(spark, "2025T1", _xlsx(tmp_path / "a.xlsx", FILAS),
                             directorio_tmp=tmp_path / "t")
    assert df.filter("salario_mensual IS NULL").count() == 1


def test_union_por_nombre_con_columnas_extra_y_otro_orden(spark, tmp_path):
    a = _xlsx(tmp_path / "a.xlsx", FILAS)
    b = _xlsx(tmp_path / "b.xlsx", FILAS, extra=["OTRA1", "OTRA2"], invertir=True)
    dfa = carga.cargar_archivo(spark, "2025T3", a, directorio_tmp=tmp_path / "t")
    dfb = carga.cargar_archivo(spark, "2025T4", b, directorio_tmp=tmp_path / "t")
    union = carga.unir_periodos([dfa, dfb])
    assert union.count() == 4
    assert union.columns == dfa.columns
    assert union.filter("periodo_archivo = '2025T4'").filter("edad = 30").count() == 1


def test_columnas_del_archivo_lee_solo_encabezado(tmp_path):
    ruta = _xlsx(tmp_path / "b.xlsx", FILAS, extra=["OTRA1"])
    assert "OTRA1" in carga.columnas_del_archivo(ruta)
    assert len(carga.columnas_del_archivo(ruta)) == len(config.COLUMNAS_ORIGINALES) + 1


def test_falta_columna_requerida(tmp_path):
    pd.DataFrame({"P05D01": ["1"]}).to_excel(tmp_path / "c.xlsx", index=False)
    with pytest.raises(ValueError):
        carga.leer_excel(tmp_path / "c.xlsx")


def test_ruta_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        carga.ruta_archivo("2025T1", tmp_path)
