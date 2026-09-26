"""Pruebas de src.random_forest."""
import os
import sys
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import functions as F

from src import config, modelado, random_forest

COLUMNAS = ["periodo_archivo", "salario_mensual", "edad", "antiguedad", "horas_semanales",
            "nivel_educativo", "categoria_ocupacional", "dominio"]
TIPOS = [pa.string(), pa.float64(), pa.float64(), pa.float64(), pa.float64(),
         pa.string(), pa.string(), pa.string()]
CONFIGS = {"arboles_5_prof_2": (5, 2), "arboles_10_prof_5": (10, 5)}


def a_spark(spark, filas, directorio):
    """DataFrame de Spark con tipos explicitos, via Parquet (sin workers de Python)."""
    tabla = pa.table({c: pa.array([f[i] for f in filas], type=t)
                      for i, (c, t) in enumerate(zip(COLUMNAS, TIPOS))})
    destino = directorio / f"{uuid.uuid4().hex}.parquet"
    pq.write_table(tabla, destino)
    return spark.read.parquet(str(destino))


def _filas(periodo, n, rng, nivel_extra=None):
    """Salario con una interaccion (la antiguedad solo paga en niveles educativos altos)."""
    filas = []
    for _ in range(n):
        edad = float(rng.integers(18, 60))
        antig = float(rng.integers(0, 15))
        horas = float(rng.integers(20, 60))
        nivel = int(rng.integers(0, 4))
        salario = 1500 + 20 * horas + 600 * nivel + (300 * antig if nivel >= 2 else 0) + rng.normal(0, 50)
        filas.append((periodo, float(salario), edad, antig, horas, str(nivel),
                      str(rng.integers(1, 5)), str(rng.integers(1, 4))))
    if nivel_extra:
        filas.append((periodo, 3000.0, 30.0, 3.0, 40.0, nivel_extra, "2", "1"))
    return filas


@pytest.fixture(scope="module")
def conjuntos(spark, tmp_path_factory):
    rng = np.random.default_rng(7)
    filas = []
    for periodo in config.PERIODOS_TRAIN:
        filas += _filas(periodo, 150, rng, nivel_extra="9" if periodo == "2025T4" else None)
    return modelado.dividir_2025(a_spark(spark, filas, tmp_path_factory.mktemp("rf")))


@pytest.fixture(scope="module")
def evaluacion(conjuntos):
    entrenamiento, validacion = conjuntos  # validacion trae el nivel "9", ausente en entrenamiento
    return random_forest.evaluar_configuraciones(entrenamiento, validacion, CONFIGS)


def test_pipeline_sin_estandarizacion_y_con_semilla(spark):
    etapas = random_forest.pipeline_random_forest(10, 5).getStages()
    nombres = [type(e).__name__ for e in etapas]
    assert nombres[-1] == "RandomForestRegressor"
    assert "StandardScaler" not in nombres
    bosque = etapas[-1]
    assert bosque.getNumTrees() == 10 and bosque.getMaxDepth() == 5
    assert bosque.getSeed() == config.SEMILLA


def test_supera_la_referencia_y_tolera_categorias_nuevas(conjuntos, evaluacion):
    entrenamiento, validacion = conjuntos
    tabla, modelos = evaluacion
    assert set(tabla["configuracion"]) == set(CONFIGS)
    assert set(modelos) == set(CONFIGS)
    referencia = modelado.metricas(modelado.predecir_referencia(
        validacion, modelado.referencia(entrenamiento)))
    mejor = random_forest.elegir_mejor(tabla)
    fila = tabla.set_index("configuracion").loc[mejor]
    assert fila["rmse_validacion"] == tabla["rmse_validacion"].min()
    assert fila["rmse_validacion"] < referencia["rmse"]
    assert fila["r2_validacion"] > 0.5
    assert modelos[mejor].transform(validacion).count() == validacion.count()


def test_misma_semilla_mismas_predicciones(conjuntos):
    entrenamiento, validacion = conjuntos
    a = random_forest.pipeline_random_forest(5, 3).fit(entrenamiento).transform(validacion)
    b = random_forest.pipeline_random_forest(5, 3).fit(entrenamiento).transform(validacion)
    assert [r[0] for r in a.select(modelado.PREDICCION).collect()] == pytest.approx(
        [r[0] for r in b.select(modelado.PREDICCION).collect()])


def test_importancias(evaluacion):
    tabla, modelos = evaluacion
    modelo = modelos[random_forest.elegir_mejor(tabla)]
    imp = random_forest.importancias(modelo)
    assert len(imp) == len(modelado.nombres_features(modelo))
    assert imp["importancia"].sum() == pytest.approx(1.0)
    assert imp["importancia"].is_monotonic_decreasing
    por_predictor = random_forest.importancias_por_predictor(imp)
    assert set(por_predictor["predictor"]) == set(modelado.PREDICTORES)
    assert por_predictor["importancia"].sum() == pytest.approx(1.0)
    # En los datos simulados el salario depende de educacion, antiguedad y horas, no de la edad.
    peso = por_predictor.set_index("predictor")["importancia"]
    assert peso["nivel_educativo"] > peso["edad"]


def test_error_por_tramo(spark, tmp_path):
    reales = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    filas = [("2025T4", r, 30.0, 1.0, 40.0, "1", "2", "1") for r in reales]
    pred = a_spark(spark, filas, tmp_path).withColumn(modelado.PREDICCION, F.lit(350.0))
    tramos = random_forest.error_por_tramo(pred, [250.0, 450.0])
    assert tramos["n"].tolist() == [2, 2, 2]
    assert tramos["n"].sum() == len(reales)
    # Residuo = real - predicho: tramo bajo sobreestimado (negativo), tramo alto subestimado (positivo).
    assert tramos["error_medio"].tolist() == pytest.approx([-200.0, 0.0, 200.0])
    assert tramos["mae"].tolist() == pytest.approx([200.0, 50.0, 200.0])
    assert tramos["rango"].tolist() == ["<= Q250", "Q250 - Q450", "> Q450"]


@pytest.mark.skipif(sys.platform == "win32" and not os.environ.get("HADOOP_HOME"),
                    reason="guardar modelos de Spark en Windows requiere winutils (HADOOP_HOME)")
def test_guardar_y_cargar_modelo(conjuntos, monkeypatch, tmp_path):
    entrenamiento, validacion = conjuntos
    monkeypatch.setattr(config, "MODELOS", tmp_path)
    modelo = random_forest.pipeline_random_forest(5, 3).fit(entrenamiento)
    modelado.guardar_modelo(modelo, "rf_prueba")
    recargado = modelado.cargar_modelo("rf_prueba")
    original = modelo.transform(validacion).select(modelado.PREDICCION).collect()
    copia = recargado.transform(validacion).select(modelado.PREDICCION).collect()
    assert [r[0] for r in original] == pytest.approx([r[0] for r in copia])
