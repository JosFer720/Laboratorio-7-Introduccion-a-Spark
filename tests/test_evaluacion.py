"""Pruebas de src.evaluacion."""
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import functions as F

from src import config, evaluacion, modelado

COLUMNAS = ["periodo_archivo", "NUM_HOGAR", "NUM_PERSONA", "salario_mensual", "edad", "antiguedad",
            "horas_semanales", "nivel_educativo", "categoria_ocupacional", "dominio"]
TIPOS = [pa.string(), pa.string(), pa.string(), pa.float64(), pa.float64(), pa.float64(),
         pa.float64(), pa.string(), pa.string(), pa.string()]


def a_spark(spark, filas, directorio):
    """DataFrame de Spark con tipos explicitos, via Parquet (sin workers de Python)."""
    tabla = pa.table({c: pa.array([f[i] for f in filas], type=t)
                      for i, (c, t) in enumerate(zip(COLUMNAS, TIPOS))})
    destino = directorio / f"{uuid.uuid4().hex}.parquet"
    pq.write_table(tabla, destino)
    return spark.read.parquet(str(destino))


def _filas(periodo, n, rng, nivel_extra=None):
    filas = []
    for i in range(n):
        edad = float(rng.integers(18, 60))
        antig = float(rng.integers(0, 15))
        horas = float(rng.integers(20, 60))
        nivel = int(rng.integers(0, 4))
        salario = 800 + 30 * edad + 80 * antig + 20 * horas + 500 * nivel + rng.normal(0, 60)
        filas.append((periodo, str(i), "1", float(salario), edad, antig, horas, str(nivel),
                      str(rng.integers(1, 5)), str(rng.integers(1, 4))))
    if nivel_extra:
        filas.append((periodo, str(n), "1", 3000.0, 30.0, 3.0, 40.0, nivel_extra, "2", "1"))
    return filas


@pytest.fixture(scope="module")
def conjuntos(spark, tmp_path_factory):
    rng = np.random.default_rng(3)
    directorio = tmp_path_factory.mktemp("eval")
    desarrollo = []
    for periodo in config.PERIODOS_TRAIN:
        desarrollo += _filas(periodo, 100, rng)
    prueba = _filas(config.PERIODO_TEST, 120, rng, nivel_extra="7")  # "7" no aparece en 2025
    return a_spark(spark, desarrollo, directorio), a_spark(spark, prueba, directorio)


@pytest.fixture(scope="module")
def predicciones(conjuntos):
    desarrollo, prueba = conjuntos
    modelos = evaluacion.entrenar_finales(desarrollo, (0.0, 0.0), (5, 3))
    return evaluacion.predecir(modelos, prueba, modelado.referencia(desarrollo)).persist()


def test_parametros_elegidos_por_menor_rmse_de_validacion():
    lineal = pd.DataFrame({"configuracion": ["a", "b"], "regParam": [0.0, 0.1],
                           "elasticNetParam": [0.0, 1.0], "rmse_validacion": [10.0, 9.0]})
    bosque = pd.DataFrame({"configuracion": ["x", "y"], "numTrees": [20, 100],
                           "maxDepth": [5, 20], "rmse_validacion": [8.0, 9.0]})
    assert evaluacion.parametros_lineal(lineal) == ("b", 0.1, 1.0)
    assert evaluacion.parametros_random_forest(bosque) == ("x", 20, 5)


def test_parametros_desde_las_tablas_guardadas():
    nombre, reg_param, elastic = evaluacion.parametros_lineal()
    assert nombre in pd.read_csv(config.TABLES / "05_metricas_regresion_lineal.csv")["configuracion"].tolist()
    nombre, arboles, profundidad = evaluacion.parametros_random_forest()
    assert arboles > 0 and profundidad > 0


def test_entrenamiento_final_rechaza_2026(conjuntos):
    desarrollo, prueba = conjuntos
    with pytest.raises(ValueError):
        evaluacion.entrenar_finales(desarrollo.unionByName(prueba), (0.0, 0.0), (5, 3))


def test_mismos_registros_para_ambos_modelos(conjuntos, predicciones):
    _, prueba = conjuntos
    assert predicciones.count() == prueba.count()
    columnas = list(evaluacion.PREDICCIONES.values()) + list(evaluacion.RESIDUOS.values())
    nulos = predicciones.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in columnas]).first()
    assert all(v == 0 for v in nulos)
    # Las claves de prueba se conservan una sola vez, sin uniones que dupliquen o pierdan filas.
    assert predicciones.select(*config.CLAVE).distinct().count() == prueba.count()


def test_residuo_real_menos_predicho(predicciones):
    for m in evaluacion.MODELOS:
        diferencia = predicciones.select(F.max(F.abs(
            F.col(evaluacion.RESIDUOS[m]) - (F.col(modelado.OBJETIVO) - F.col(evaluacion.PREDICCIONES[m])))))
        assert diferencia.first()[0] == pytest.approx(0.0, abs=1e-9)


def test_metricas_de_prueba(predicciones):
    tabla = evaluacion.metricas_prueba(predicciones).set_index("modelo")
    assert list(tabla.index) == ["referencia", "regresión lineal", "random forest"]
    assert (tabla["n"] == predicciones.count()).all()
    assert tabla.loc["regresión lineal", "rmse"] < tabla.loc["referencia", "rmse"]
    assert tabla.loc["random forest", "rmse"] < tabla.loc["referencia", "rmse"]
    assert tabla.loc["regresión lineal", "r2"] > 0.8


def test_muestra_comun(predicciones):
    muestra = evaluacion.muestra_comun(predicciones, n=50)
    assert 0 < len(muestra) <= 50
    for m in evaluacion.MODELOS:
        assert muestra[evaluacion.PREDICCIONES[m]].notna().all()
        assert np.allclose(muestra[evaluacion.RESIDUOS[m]],
                           muestra[modelado.OBJETIVO] - muestra[evaluacion.PREDICCIONES[m]])


def test_error_por_grupo_con_valores_conocidos(spark, tmp_path):
    filas = [("2026T1", str(i), "1", real, 30.0, 1.0, 40.0, nivel, "2", "1")
             for i, (real, nivel) in enumerate([(100.0, "0"), (300.0, "0"), (500.0, "5")])]
    pred = a_spark(spark, filas, tmp_path)
    pred = (pred.withColumn(evaluacion.PREDICCIONES["regresion_lineal"], F.lit(200.0))
            .withColumn(evaluacion.PREDICCIONES["random_forest"], F.col(modelado.OBJETIVO)))
    for m in evaluacion.MODELOS:
        pred = pred.withColumn(evaluacion.RESIDUOS[m], F.col(modelado.OBJETIVO) - F.col(evaluacion.PREDICCIONES[m]))
    tabla = evaluacion.error_por_grupo(pred, "nivel_educativo")
    assert tabla["codigo"].tolist() == ["0", "5"]
    assert tabla["categoria"].tolist() == ["ninguno", "superior"]
    assert tabla["n"].tolist() == [2, 1]
    assert tabla["mae_regresion_lineal"].tolist() == pytest.approx([100.0, 300.0])
    assert tabla["error_medio_regresion_lineal"].tolist() == pytest.approx([0.0, 300.0])
    assert tabla["mae_random_forest"].tolist() == pytest.approx([0.0, 0.0])


def test_percentiles_y_rango(spark, tmp_path, predicciones):
    filas = [("2026T1", str(i), "1", float(v), 30.0, 1.0, 40.0, "1", "2", "1") for i, v in enumerate(range(1, 101))]
    df = a_spark(spark, filas, tmp_path)
    tabla = evaluacion.percentiles_salario({"a": df, "b": df}, [0.5, 0.9])
    assert tabla.loc["n", "a"] == 100
    assert tabla.loc["p50", "a"] == pytest.approx(50.0)
    assert tabla.loc["p90", "b"] == pytest.approx(90.0)
    rango = evaluacion.rango_predicciones(predicciones)
    assert list(rango.index) == ["salario real", "regresión lineal", "random forest"]
    assert (rango["minimo"] <= rango["maximo"]).all()


def _pred_conocida(spark, tmp_path, reales, predichos):
    filas = [("2026T1", str(i), "1", float(r), 30.0, 1.0, 40.0, "1", "2", "1") for i, r in enumerate(reales)]
    pred = a_spark(spark, filas, tmp_path)
    for m in evaluacion.MODELOS:
        valores = F.create_map(*[x for i, p in enumerate(predichos) for x in (F.lit(str(i)), F.lit(float(p)))])
        pred = (pred.withColumn(evaluacion.PREDICCIONES[m], valores[F.col("NUM_HOGAR")])
                .withColumn(evaluacion.RESIDUOS[m], F.col(modelado.OBJETIVO) - F.col(evaluacion.PREDICCIONES[m])))
    return pred


def test_resumen_errores_con_valores_conocidos(spark, tmp_path):
    # Residuos: -50, +10, 0, +600 (el ultimo es un salario alto subestimado).
    pred = _pred_conocida(spark, tmp_path, [100, 200, 300, 1000], [150, 190, 300, 400])
    tabla = evaluacion.resumen_errores(pred, umbral_alto=500, umbral_cola=500).set_index("modelo")
    fila = tabla.loc["random forest"]
    assert fila["pct_error_hasta_20pct"] == pytest.approx(50.0)  # 10/200 y 0/300
    assert fila["pct_subestimados"] == pytest.approx(50.0)
    assert fila["pct_subestimados_salario_alto"] == pytest.approx(100.0)
    assert fila["pct_error_cuadratico_cola"] == pytest.approx(100 * 360000 / (2500 + 100 + 360000))


def test_error_por_decil_prediccion(spark, tmp_path):
    pred = _pred_conocida(spark, tmp_path, list(range(10, 210, 10)), list(range(10, 210, 10)))
    tabla = evaluacion.error_por_decil_prediccion(pred, "regresion_lineal", grupos=4)
    assert tabla["n"].sum() == 20
    assert len(tabla) == 4
    assert tabla["prediccion_media"].is_monotonic_increasing
    assert tabla["error_medio"].tolist() == pytest.approx([0.0] * 4)


def test_error_por_tramo_para_ambos_modelos(predicciones):
    cortes = evaluacion.cortes_tramos(predicciones, [0.5])
    tramos = evaluacion.error_por_tramo(predicciones, cortes)
    assert set(tramos["modelo"]) == {"regresión lineal", "random forest"}
    for _, grupo in tramos.groupby("modelo"):
        assert grupo["n"].sum() == predicciones.count()
