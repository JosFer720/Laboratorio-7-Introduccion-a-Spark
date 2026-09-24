"""Pruebas de src.segmentacion."""
import numpy as np
import pandas as pd
import pytest

from src import carga, segmentacion


@pytest.fixture(scope="module")
def datos(spark, tmp_path_factory):
    """Tres grupos bien separados: jovenes con jornada larga, mayores con antiguedad alta y jornada corta."""
    rng = np.random.default_rng(0)
    grupos = [
        dict(edad=22, antiguedad=1, horas_semanales=60, salario_mensual=2500, n=60),
        dict(edad=55, antiguedad=25, horas_semanales=40, salario_mensual=8000, n=50),
        dict(edad=35, antiguedad=5, horas_semanales=15, salario_mensual=1200, n=40),
    ]
    filas = []
    for g in grupos:
        for _ in range(g["n"]):
            filas.append({
                "edad": g["edad"] + rng.normal(0, 1),
                "antiguedad": max(0.0, g["antiguedad"] + rng.normal(0, 0.5)),
                "horas_semanales": g["horas_semanales"] + rng.normal(0, 1),
                "salario_mensual": g["salario_mensual"] * float(np.exp(rng.normal(0, 0.05))),
                "nivel_educativo": str(rng.integers(0, 3)),
                "categoria_ocupacional": "2",
            })
    pdf = pd.DataFrame(filas)
    df = carga.a_spark_texto(spark, pdf.astype(str), tmp_path_factory.mktemp("seg"))
    for c in ["edad", "antiguedad", "horas_semanales", "salario_mensual"]:
        df = df.withColumn(c, df[c].cast("double"))
    return segmentacion.agregar_salario_log(df)


@pytest.fixture(scope="module")
def evaluacion(datos):
    return segmentacion.evaluar_k(datos, segmentacion.VARIABLES_PERFIL, ks=[2, 3, 4])


def test_salario_log(datos):
    fila = datos.select("salario_mensual", segmentacion.SALARIO_LOG).first()
    assert fila[1] == pytest.approx(np.log10(fila[0]))


def test_evaluar_k_metricas(evaluacion):
    tabla, modelos = evaluacion
    assert list(tabla["k"]) == [2, 3, 4] and set(modelos) == {2, 3, 4}
    assert all(sum(t) == 150 for t in tabla["tamanos"])
    assert tabla["wssse"].is_monotonic_decreasing
    assert tabla["silhouette"].between(-1, 1).all()
    assert np.isnan(tabla["wssse_reduccion_pct"].iloc[0])


def test_elegir_k_encuentra_los_tres_grupos(evaluacion):
    tabla, modelos = evaluacion
    assert segmentacion.elegir_k(tabla) == 3
    assert sorted(tabla.set_index("k").loc[3, "tamanos"]) == [40, 50, 60]


def test_elegir_k_codo_con_tamano_minimo():
    tabla = pd.DataFrame({"k": [2, 3, 4, 5], "silhouette": [0.6, 0.4, 0.5, 0.5],
                          "cluster_menor_pct": [30.0, 12.0, 10.0, 1.0],
                          "wssse_reduccion_pct": [np.nan, 20.0, 25.0, 30.0]})
    assert segmentacion.elegir_k(tabla) == 4
    assert segmentacion.elegir_k(tabla, tamano_minimo_pct=11) == 3
    assert segmentacion.elegir_k(tabla, tamano_minimo_pct=50) == 2


def test_centroides_en_unidades_originales(datos, evaluacion):
    _, modelos = evaluacion
    cent = segmentacion.centroides(modelos[3], segmentacion.VARIABLES_PERFIL)
    assert sorted(cent["edad"].round()) == [22, 35, 55]
    assert abs(cent["edad_z"].mean()) < 0.5


def test_perfil_composicion_y_nombres(datos, evaluacion):
    _, modelos = evaluacion
    pred = modelos[3].transform(datos)
    perfil = segmentacion.perfil_clusters(pred)
    assert perfil["registros"].sum() == 150
    assert perfil["porcentaje"].sum() == pytest.approx(100.0)
    comp = segmentacion.composicion(pred, "nivel_educativo")
    assert np.allclose(comp.sum(axis=1), 100.0)
    assert set(segmentacion.predominante(comp).columns) == {"categoria", "porcentaje"}

    nombres = segmentacion.nombrar_clusters(segmentacion.centroides(modelos[3], segmentacion.VARIABLES_PERFIL))
    cent = segmentacion.centroides(modelos[3], segmentacion.VARIABLES_PERFIL)
    joven = cent["edad"].idxmin()
    mayor = cent["edad"].idxmax()
    assert "Jóvenes" in nombres[joven] and "jornada larga" in nombres[joven]
    assert "Mayores" in nombres[mayor] and "antigüedad alta" in nombres[mayor]


def test_nombre_promedio():
    cent = pd.DataFrame({"edad_z": [0.1], "antiguedad_z": [-0.2], "horas_semanales_z": [0.3]})
    assert segmentacion.nombrar_clusters(cent)[0] == "Perfil promedio"


def test_proyeccion_pca_muestra(datos, evaluacion):
    _, modelos = evaluacion
    pred = modelos[3].transform(datos)
    muestra, varianza = segmentacion.proyeccion_pca(pred, n=100)
    assert len(muestra) <= 100
    assert {"pc1", "pc2", "cluster"} <= set(muestra.columns)
    assert len(varianza) == 2 and sum(varianza) <= 1.0 + 1e-9
