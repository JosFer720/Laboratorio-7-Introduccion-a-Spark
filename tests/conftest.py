"""Fixtures compartidos de las pruebas."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def spark():
    from src.config import crear_spark

    sesion = crear_spark("pruebas")
    yield sesion
    sesion.stop()
