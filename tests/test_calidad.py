"""Pruebas de src.calidad."""
import pandas as pd
import pytest

from src import calidad, carga, config

BASE = {
    "P05D01": "3000", "P02A03": "30", "P05C07A": "2", "P05C07B": "6", "P05H01A": "40",
    "P03A03A": "0", "P05C16": "2", "DOMINIO": "1", "OCUPADOS": "1",
    "NUM_HOGAR": "1", "NUM_PERSONA": "1", "FACTOR": "100", "ANIO": "2025", "TRIMESTRE": "2",
}


def _df(spark, tmp_path, filas, periodo="2025T1"):
    pdf = pd.DataFrame([{**BASE, **f} for f in filas], columns=config.COLUMNAS_ORIGINALES)
    df = carga.homologar_tipos(carga.a_spark_texto(spark, pdf, tmp_path / f"t_{periodo}_{len(filas)}"))
    return carga.agregar_procedencia(df, periodo, "x.xlsx")


def _fila(i, **kw):
    return {"NUM_HOGAR": str(i), **kw}


def test_embudo_orden_fijo_y_conteos(spark, tmp_path):
    filas = [
        _fila(1),                                   # valido
        _fila(2, P02A03="14", P05C07A="1"),         # < 15 anios
        _fila(3, OCUPADOS="2"),                     # no ocupado
        _fila(4, P05C16="5"),                       # no asalariado
        _fila(5, P05D01="0"),                       # salario no positivo
        _fila(6, P05D01=None),                      # salario faltante
        _fila(7, P05C07A="40"),                     # antiguedad > edad
        _fila(8, P05C07B="12"),                     # meses fuera de 0-11
        _fila(9, P05H01A="169"),                    # horas > 168
        _fila(10, P05H01A="0"),                     # horas = 0
    ]
    tabla = calidad.embudo(_df(spark, tmp_path, filas))
    total = tabla[tabla["periodo_archivo"] == "TOTAL"].set_index("paso")
    assert total.loc["registros originales", "restantes"] == 10
    assert total.loc["edad finita y >= 15", "excluidos"] == 1
    assert total.loc["ocupado (OCUPADOS = 1)", "excluidos"] == 1
    assert total.loc["asalariado (P05C16 en 1-4)", "excluidos"] == 1
    assert total.loc["salario finito y > 0", "excluidos"] == 2
    assert total.loc["antiguedad valida (meses 0-11, >= 0 y <= edad)", "excluidos"] == 2
    assert total.loc["horas habituales > 0 y <= 168", "excluidos"] == 2
    assert total["excluidos"].sum() == 9
    assert total.iloc[-1]["restantes"] == 1


def test_preparar_devuelve_columnas_finales_y_antiguedad(spark, tmp_path):
    df = calidad.preparar(_df(spark, tmp_path, [_fila(1)]))
    assert df.columns == config.COLUMNAS_PREPARADAS
    fila = df.first()
    assert fila["antiguedad"] == pytest.approx(2.5)
    assert fila["nivel_educativo"] == "0"        # cero = ninguno, no faltante


def test_categoricas_desconocidas(spark, tmp_path):
    filas = [_fila(1, P03A03A=None), _fila(2, P03A03A="77"), _fila(3, DOMINIO="9"), _fila(4)]
    df = calidad.preparar(_df(spark, tmp_path, filas)).orderBy("NUM_HOGAR")
    valores = {r["NUM_HOGAR"]: (r["nivel_educativo"], r["dominio"]) for r in df.collect()}
    assert valores["1"] == (config.DESCONOCIDO, "1")
    assert valores["2"] == (config.DESCONOCIDO, "1")
    assert valores["3"] == ("0", config.DESCONOCIDO)
    assert valores["4"] == ("0", "1")


def test_faltantes(spark, tmp_path):
    df = _df(spark, tmp_path, [_fila(1), _fila(2, P05D01=None), _fila(3, P05D01=None, P02A03=None)])
    tabla = calidad.faltantes(df, ["salario_mensual", "edad"]).set_index("variable")
    assert tabla.loc["salario_mensual", "faltantes"] == 2
    assert tabla.loc["edad", "faltantes"] == 1
    assert tabla.loc["salario_mensual", "porcentaje"] == pytest.approx(66.67)


def test_duplicados_exactos_y_conflictos(spark, tmp_path):
    filas = [
        _fila(1), _fila(1),                         # repeticion exacta
        _fila(2), _fila(2, P05D01="5000"),          # conflicto en salario
        _fila(3),                                   # unica
    ]
    df = _df(spark, tmp_path, filas)
    resumen = calidad.resumen_unicidad(df)
    assert resumen["registros"] == 5
    assert resumen["claves_distintas"] == 3
    assert resumen["claves_duplicadas"] == 2
    assert resumen["filas_en_claves_duplicadas"] == 4
    detalle = {r["NUM_HOGAR"]: r["tipo"] for r in calidad.detalle_duplicados(df).collect()}
    assert detalle == {"1": "repeticion exacta", "2": "conflicto"}
    conflictos = calidad.columnas_en_conflicto(df).set_index("columna")["claves_con_diferencia"]
    assert conflictos["salario_mensual"] == 1
    assert conflictos["edad"] == 0
    assert df.count() == 5                          # no se elimino nada


def test_misma_persona_en_dos_periodos_no_es_duplicado(spark, tmp_path):
    a = _df(spark, tmp_path, [_fila(1)], "2025T1")
    b = _df(spark, tmp_path, [_fila(1)], "2025T2")
    union = carga.unir_periodos([a, b])
    assert calidad.resumen_unicidad(union)["claves_duplicadas"] == 0


def test_resumen_por_archivo(spark, tmp_path):
    df = _df(spark, tmp_path, [_fila(1), _fila(2, OCUPADOS="2")])
    resumen = calidad.resumen_por_archivo(calidad.embudo(df)).set_index("periodo_archivo")
    assert resumen.loc["2025T1", "antes"] == 2
    assert resumen.loc["2025T1", "despues"] == 1
    assert resumen.loc["2025T1", "retenido_pct"] == 50.0


def test_codigos_observados_marca_no_reconocidos(spark, tmp_path):
    df = _df(spark, tmp_path, [_fila(1), _fila(2, DOMINIO="9"), _fila(3, DOMINIO=None)])
    tabla = calidad.codigos_observados(df, "dominio").set_index("codigo")
    assert bool(tabla.loc["1", "reconocido"]) and not bool(tabla.loc["9", "reconocido"])
    assert not bool(tabla.loc["<AUSENTE>", "reconocido"])
