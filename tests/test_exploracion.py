"""Pruebas de src.exploracion."""
import numpy as np
import pandas as pd
import pytest

from src import carga, exploracion


def _df(spark, tmp_path, pdf):
    """DataFrame de Spark con tipos numericos a partir de un DataFrame de pandas."""
    texto = pdf.astype(str)
    df = carga.a_spark_texto(spark, texto, tmp_path / "t")
    for c in pdf.columns:
        if pd.api.types.is_numeric_dtype(pdf[c]):
            df = df.withColumn(c, df[c].cast("double"))
    return df


@pytest.fixture
def datos(spark, tmp_path):
    pdf = pd.DataFrame({
        "periodo_archivo": ["2025T1"] * 4 + ["2025T2"] * 4,
        "salario_mensual": [1000.0, 2000.0, 3000.0, 10000.0, 1500.0, 2500.0, 3500.0, 4500.0],
        "edad": [20.0, 30.0, 40.0, 50.0, 25.0, 35.0, 45.0, 55.0],
        "antiguedad": [1.0, 5.0, 10.0, 20.0, 2.0, 6.0, 12.0, 22.0],
        "horas_semanales": [40.0, 44.0, 48.0, 40.0, 36.0, 40.0, 44.0, 48.0],
        "nivel_educativo": ["0", "1", "1", "DESCONOCIDO", "0", "2", "2", "1"],
    })
    return pdf, _df(spark, tmp_path, pdf)


def test_descriptivas_coinciden_con_pandas(datos):
    pdf, df = datos
    tabla = exploracion.descriptivas(df)
    s = pdf["salario_mensual"]
    fila = tabla.loc["salario_mensual"]
    assert fila["n"] == 8
    assert fila["media"] == pytest.approx(s.mean())
    assert fila["mediana"] == pytest.approx(s.median())
    assert fila["desv_estandar"] == pytest.approx(s.std())
    assert fila["minimo"] == s.min() and fila["maximo"] == s.max()
    assert fila["p25"] == pytest.approx(np.percentile(s, 25))
    assert fila["p75"] == pytest.approx(np.percentile(s, 75))
    assert fila["p95"] == pytest.approx(np.percentile(s, 95))
    assert list(tabla.index) == exploracion.VARIABLES_NUMERICAS


def test_asimetria_positiva_con_cola_derecha(datos):
    _, df = datos
    r = exploracion.asimetria(df, "salario_mensual")
    assert r["media"] > r["mediana"]
    assert r["asimetria"] > 0


def test_conteo_categorias_y_orden(datos):
    _, df = datos
    tabla = exploracion.conteo_categorias(df, "nivel_educativo")
    assert list(tabla["categoria"]) == ["0", "1", "2", "DESCONOCIDO"]
    assert tabla["registros"].sum() == 8
    assert tabla["porcentaje"].sum() == pytest.approx(100.0)


def test_salario_por_grupo(datos):
    _, df = datos
    tabla = exploracion.salario_por_grupo(df, "nivel_educativo").set_index("categoria")
    assert tabla.loc["0", "registros"] == 2
    assert tabla.loc["0", "salario_mediano"] == pytest.approx(1250.0)


def test_resumen_por_trimestre(datos):
    _, df = datos
    tabla = exploracion.resumen_por_trimestre(df).set_index("periodo_archivo")
    assert list(tabla["registros"]) == [4, 4]
    assert tabla.loc["2025T1", "salario_mediano"] == pytest.approx(2500.0)


def test_histograma_suma_todos_los_registros(datos):
    _, df = datos
    for log in (False, True):
        h = exploracion.histograma(df, "salario_mensual", bins=5, log10=log)
        assert h["registros"].sum() == 8
        assert len(h) == 5
    assert exploracion.histograma(df, "salario_mensual", bins=5, log10=True)["hasta"].max() == pytest.approx(4.0)


def test_muestra_respeta_el_maximo(datos):
    _, df = datos
    assert len(exploracion.muestra(df, n=3)) <= 3


def test_correlacion_coincide_con_pandas(datos):
    pdf, df = datos
    m = exploracion.matriz_correlacion(df)
    esperado = pdf[exploracion.VARIABLES_NUMERICAS].corr()
    assert m.shape == (4, 4)
    assert np.allclose(m.values, esperado.values, atol=1e-9)
    assert np.allclose(np.diag(m.values), 1.0)
    assert list(m.index) == list(m.columns) == exploracion.VARIABLES_NUMERICAS
