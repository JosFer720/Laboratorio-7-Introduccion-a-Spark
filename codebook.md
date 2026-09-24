# Codebook — Laboratorio 7 (ENEIC, bases de Personas)

Fuente: Encuesta Nacional de Empleo e Ingresos Continua (ENEIC), Instituto Nacional de Estadística de Guatemala. Solo se usan las bases de **Personas**. Los códigos deben validarse contra el diccionario de datos que acompaña a cada base.

## Archivos y periodos

| `periodo_archivo` | `anio_archivo` | `trimestre_calendario` | Valores originales de `TRIMESTRE` | Uso |
|---|---|---|---|---|
| 2025T1 | 2025 | 1 | 2 | Desarrollo |
| 2025T2 | 2025 | 2 | 3 (y 2 en 175 registros) | Desarrollo |
| 2025T3 | 2025 | 3 | 4 | Desarrollo |
| 2025T4 | 2025 | 4 | 5 | Desarrollo (302 columnas originales) |
| 2026T1 | 2026 | 1 | 6 | Prueba final |

`TRIMESTRE` se conserva tal como llegó y no se usa como trimestre calendario; el periodo se asigna por el archivo de procedencia.

## Variables originales seleccionadas

| Original | Nombre analítico | Tipo | Descripción |
|---|---|---|---|
| `P05D01` | `salario_mensual` | double | Sueldo o salario mensual sin descuentos de la ocupación principal, en quetzales. Variable objetivo. |
| `P02A03` | `edad` | double | Edad en años. |
| `P05C07A` | `antiguedad_anios` | double | Años de antigüedad en la ocupación. |
| `P05C07B` | `antiguedad_meses` | double | Meses de antigüedad (componente entero de 0 a 11). |
| `P05H01A` | `horas_semanales` | double | Horas habituales de trabajo por semana en la ocupación principal. |
| `P03A03A` | `nivel_educativo` | texto | Nivel educativo. El código `0` es "ninguno", no un faltante. |
| `P05C16` | `categoria_ocupacional` | texto | Categoría ocupacional: 1 empleado de gobierno, 2 empleado de empresa privada, 3 empleado jornalero o peón, 4 servicio doméstico. |
| `DOMINIO` | `dominio` | texto | Dominio de estudio. |
| `OCUPADOS` | `ocupado` | int | 1 = persona ocupada. |
| `NUM_HOGAR` | `NUM_HOGAR` | texto | Identificador de hogar (auditoría). |
| `NUM_PERSONA` | `NUM_PERSONA` | texto | Identificador de persona dentro del hogar (auditoría). |
| `FACTOR` | `FACTOR` | double | Factor de expansión del diseño muestral. Se conserva; no se usa en clustering, modelos ni métricas. |
| `ANIO` | `ANIO` | int | Año original de la fuente. |
| `TRIMESTRE` | `TRIMESTRE` | int | Trimestre original de la fuente. |

Los códigos (`NUM_HOGAR`, `NUM_PERSONA`, `nivel_educativo`, `categoria_ocupacional`, `dominio`) se guardan como texto canónico: `1`, `1.0` y ` 1 ` se representan como `1`.

## Variables derivadas

| Variable | Definición |
|---|---|
| `periodo_archivo` | Corte publicado al que pertenece el archivo (por ejemplo `2025T1`). |
| `anio_archivo` | Año del archivo. |
| `trimestre_calendario` | Trimestre calendario del archivo. |
| `archivo_origen` | Nombre del archivo de procedencia. |
| `antiguedad` | `antiguedad_anios + antiguedad_meses / 12`, en años. |

## Población analítica y orden de los filtros

1. `edad` finita y mayor o igual a 15.
2. `ocupado = 1`.
3. `categoria_ocupacional` en 1, 2, 3 o 4.
4. `salario_mensual` numérico, finito y mayor que 0.
5. Antigüedad: años no negativos, meses enteros entre 0 y 11, `antiguedad` no negativa y menor o igual a `edad`.
6. `horas_semanales` mayor que 0 y menor o igual a 168.

Los registros que no permiten evaluar un criterio se excluyen y se contabilizan. El salario no se imputa. Los salarios extremos no se eliminan.

Las variables categóricas ausentes o con códigos no reconocidos se representan como `DESCONOCIDO`.

## Conjuntos preparados (`data/processed/`)

`personas_2025.parquet` y `personas_2026.parquet`, con las columnas:

`periodo_archivo`, `anio_archivo`, `trimestre_calendario`, `archivo_origen`, `NUM_HOGAR`, `NUM_PERSONA`, `FACTOR`, `ANIO`, `TRIMESTRE`, `salario_mensual`, `edad`, `antiguedad`, `horas_semanales`, `nivel_educativo`, `categoria_ocupacional`, `dominio`.

## Predictores de los modelos supervisados

Exactamente seis: `edad`, `antiguedad`, `horas_semanales`, `nivel_educativo`, `categoria_ocupacional` y `dominio`. No se usan identificadores, `FACTOR`, otros ingresos, salario por hora derivado del objetivo ni la etiqueta de cluster.

## Clave de unicidad

`periodo_archivo` + `NUM_HOGAR` + `NUM_PERSONA`. Una misma persona puede aparecer en periodos distintos (diseño longitudinal con rotación); eso no es un duplicado.
