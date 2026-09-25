"""Pruebas de src.modelado y src.regresion_lineal."""
import os
import sys
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import functions as F

from src import config, modelado, regresion_lineal

COLUMNAS = ["periodo_archivo", "salario_mensual", "edad", "antiguedad", "horas_semanales",
            "nivel_educativo", "categoria_ocupacional", "dominio"]
TIPOS = [pa.string(), pa.float64(), pa.float64(), pa.float64(), pa.float64(),
         pa.string(), pa.string(), pa.string()]


def a_spark(spark, filas, directorio):
    """DataFrame de Spark con tipos explicitos, via Parquet (igual que src.carga, sin workers de Python)."""
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
        nivel = str(rng.integers(0, 4))
        salario = 500 + 40 * edad + 90 * antig + 25 * horas + 400 * int(nivel) + rng.normal(0, 50)
        filas.append((periodo, float(salario), edad, antig, horas, nivel,
                      str(rng.integers(1, 5)), str(rng.integers(1, 4))))
    if nivel_extra:
        filas.append((periodo, 3000.0, 30.0, 3.0, 40.0, nivel_extra, "2", "1"))
    return filas


@pytest.fixture(scope="module")
def desarrollo(spark, tmp_path_factory):
    rng = np.random.default_rng(1)
    filas = []
    for periodo in config.PERIODOS_TRAIN:
        filas += _filas(periodo, 150, rng, nivel_extra="9" if periodo == "2025T4" else None)
    return a_spark(spark, filas, tmp_path_factory.mktemp("des"))


@pytest.fixture(scope="module")
def conjuntos(desarrollo):
    return modelado.dividir_2025(desarrollo)


def test_division_por_periodo(conjuntos):
    entrenamiento, validacion = conjuntos
    assert {r[0] for r in entrenamiento.select("periodo_archivo").distinct().collect()} == {
        "2025T1", "2025T2", "2025T3"}
    assert {r[0] for r in validacion.select("periodo_archivo").distinct().collect()} == {"2025T4"}
    assert entrenamiento.count() == 450
    assert validacion.count() == 151


def test_division_rechaza_2026(spark, desarrollo, tmp_path):
    extra = a_spark(spark, [("2026T1", 3000.0, 30.0, 3.0, 40.0, "1", "2", "1")], tmp_path)
    with pytest.raises(ValueError):
        modelado.dividir_2025(desarrollo.unionByName(extra))


def test_metricas_conocidas(spark, tmp_path):
    filas = [("2025T1", real, 30.0, 1.0, 40.0, "1", "2", "1") for real in (100.0, 200.0, 300.0)]
    df = a_spark(spark, filas, tmp_path).withColumn(modelado.PREDICCION, F.lit(200.0))
    m = modelado.metricas(df)
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(200 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(20000 / 3))
    assert m["r2"] == pytest.approx(0.0, abs=1e-9)


def test_referencia_media_y_r2_cero_en_el_mismo_conjunto(conjuntos):
    entrenamiento, _ = conjuntos
    media = modelado.referencia(entrenamiento)
    esperado = entrenamiento.agg(F.avg(modelado.OBJETIVO)).first()[0]
    assert media == pytest.approx(esperado)
    m = modelado.metricas(modelado.predecir_referencia(entrenamiento, media))
    assert m["r2"] == pytest.approx(0.0, abs=1e-9)


def test_residuo_positivo_es_subestimacion(spark, tmp_path):
    df = a_spark(spark, [("2025T1", 500.0, 30.0, 1.0, 40.0, "1", "2", "1")], tmp_path)
    fila = modelado.agregar_residuo(df.withColumn(modelado.PREDICCION, F.lit(400.0))).first()
    assert fila[modelado.RESIDUO] == pytest.approx(100.0)


def test_dimension_y_nombres_de_features(conjuntos):
    entrenamiento, _ = conjuntos
    from pyspark.ml import Pipeline

    modelo = Pipeline(stages=modelado.etapas_preprocesamiento()).fit(entrenamiento)
    nombres = modelado.nombres_features(modelo)
    vector = modelo.transform(entrenamiento).select("features").first()[0]
    assert len(nombres) == vector.size
    assert nombres[:3] == config.NUMERICAS
    assert "nivel_educativo=0" in nombres


def test_regresion_lineal_supera_la_referencia_y_tolera_categorias_nuevas(conjuntos):
    entrenamiento, validacion = conjuntos  # validacion trae el nivel "9", ausente en entrenamiento
    tabla, modelos = regresion_lineal.evaluar_configuraciones(
        entrenamiento, validacion,
        {"ridge_0.01": (0.01, 0.0), "ridge_10": (10.0, 0.0), "sin_reg": (0.0, 0.0)})
    referencia = modelado.metricas(modelado.predecir_referencia(
        validacion, modelado.referencia(entrenamiento)))
    mejor = regresion_lineal.elegir_mejor(tabla)
    fila = tabla.set_index("configuracion").loc[mejor]
    assert fila["rmse_validacion"] < referencia["rmse"]
    assert fila["r2_validacion"] > 0.9
    assert fila["rmse_validacion"] == tabla["rmse_validacion"].min()
    coef = regresion_lineal.coeficientes(modelos[mejor])
    assert coef.loc[0, "variable"] == "(intercepto)"
    assert len(coef) == len(modelado.nombres_features(modelos[mejor])) + 1


@pytest.mark.skipif(sys.platform == "win32" and not os.environ.get("HADOOP_HOME"),
                    reason="guardar modelos de Spark en Windows requiere winutils (HADOOP_HOME)")
def test_guardar_y_cargar_modelo(conjuntos, monkeypatch, tmp_path):
    entrenamiento, validacion = conjuntos
    monkeypatch.setattr(config, "MODELOS", tmp_path)
    modelo = regresion_lineal.pipeline_lineal(0.01, 0.0).fit(entrenamiento)
    modelado.guardar_modelo(modelo, "lineal_prueba")
    recargado = modelado.cargar_modelo("lineal_prueba")
    original = modelo.transform(validacion).select(modelado.PREDICCION).collect()
    copia = recargado.transform(validacion).select(modelado.PREDICCION).collect()
    assert [r[0] for r in original] == pytest.approx([r[0] for r in copia])
