# 1. Carga, armonización y calidad de datos

Código: `src/carga.py`, `src/calidad.py` y `src/config.py`. Notebook: `notebooks/01_carga_armonizacion_calidad.ipynb`.

## Archivos y procedencia

| Archivo | Registros | Columnas | Uso |
|---|---|---|---|
| I de 2025 | 51,588 | 270 | entrenamiento |
| II de 2025 | 51,167 | 270 | entrenamiento |
| III de 2025 | 51,583 | 270 | entrenamiento |
| IV de 2025 | 49,338 | 302 | validación y entrenamiento final |
| I de 2026 | 49,843 | 270 | prueba final |

- Cada xlsx se lee con pandas (solo las 14 columnas requeridas y todo como texto), se pasa a Spark con tipos explícitos y se guarda en Parquet, **de uno en uno**.
- Los códigos se llevan a un texto canónico (`1`, `1.0` y ` 1 ` pasan a `1`) antes de unir.
- El archivo IV de 2025 trae 42 columnas que no existen en III y le faltan 10 de III. Por eso 295 de sus columnas cambian de posición y los archivos se unen con `unionByName`, nunca por posición.
- `periodo_archivo`, `anio_archivo`, `trimestre_calendario` y `archivo_origen` salen del archivo de procedencia. `TRIMESTRE` se conserva tal cual: va de 2 (I de 2025) a 6 (I de 2026), y el archivo II de 2025 trae 175 registros con 2 y 50,992 con 3. Restar uno no lo resolvería.

## Faltantes

Se cuentan antes de los filtros. Los porcentajes altos en salario, categoría, antigüedad y horas corresponden sobre todo a preguntas que **no aplican** (el módulo de empleo solo se pregunta a personas ocupadas). Eso es distinto de una respuesta no registrada por alguien que sí debía responder.

## Embudo de filtros (orden fijo)

| Paso | 2025 restantes | 2025 excluidos | 2026 restantes | 2026 excluidos |
|---|---|---|---|---|
| Registros originales | 203,676 | — | 49,843 | — |
| Edad finita y ≥ 15 | 140,790 | 62,886 | 35,040 | 14,803 |
| Ocupado (`OCUPADOS = 1`) | 88,222 | 52,568 | 21,734 | 13,306 |
| Asalariado (`P05C16` en 1–4) | 53,025 | 35,197 | 13,258 | 8,476 |
| Salario finito y > 0 | 53,025 | 0 | 13,258 | 0 |
| Antigüedad válida (meses enteros 0–11, ≥ 0 y ≤ edad) | 53,025 | 0 | 13,258 | 0 |
| Horas > 0 y ≤ 168 | 53,025 | 0 | 13,258 | 0 |

- Todo lo excluido son **cambios de población** (menores de 15, no ocupados y no asalariados). Los criterios de salario, antigüedad y horas no excluyen a ningún asalariado.
- Lo que no permite evaluar un criterio (nulo) se excluye y se cuenta en ese paso. El salario no se imputa.
- La población analítica es el 26.0 % de 2025 (entre 25.7 % y 26.4 % por archivo) y el 26.6 % de 2026.

## Categóricas y unicidad

- El nivel educativo se valida contra los códigos 0 a 7 del diccionario (0 = ninguno, no un faltante), la categoría ocupacional contra 1 a 4 y el dominio contra 1 a 3. Lo ausente o no reconocido pasa a `DESCONOCIDO`. En la población analítica no queda ninguno.
- La clave `periodo_archivo + NUM_HOGAR + NUM_PERSONA` es única: 203,676 combinaciones en 203,676 filas de 2025 y 49,843 en 49,843 de 2026. No hay repeticiones exactas ni conflictos, y no se usó `dropDuplicates()`.
- Una misma persona puede aparecer en varios trimestres (diseño longitudinal con rotación). Eso no es un duplicado.

## Salida

`data/processed/personas_2025.parquet` (53,025 registros) y `personas_2026.parquet` (13,258), con las 16 columnas documentadas en `codebook.md`. El análisis no es ponderado. `FACTOR` se conserva y serviría como peso de expansión para estimar a la población.
