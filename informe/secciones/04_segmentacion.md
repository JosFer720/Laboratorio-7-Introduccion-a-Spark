# 4. Segmentación de perfiles mediante KMeans

Código: `src/segmentacion.py`. Notebook: `notebooks/03_segmentacion_kmeans.ipynb`. Tablas y figuras con prefijo `03_` en `results/`.

## Variables

- **Segmentación principal (`sin_salario`):** edad, antigüedad y horas semanales habituales.
- **Contraste (`con_salario`):** las tres anteriores más `log10(salario)`.

Las variables categóricas (nivel educativo, categoría ocupacional y dominio) no entran en KMeans, porque sus códigos no son magnitudes y la distancia euclidiana entre ellos no tiene significado. Se usan después para describir cada perfil.

El salario se deja fuera de la segmentación principal por dos razones:

1. Es la variable que se estima en la segunda parte. Si define los grupos, los perfiles quedan construidos con la respuesta.
2. Sin transformar, su cola derecha haría que unos pocos salarios extremos formaran un grupo propio.

Cuando se incluye como contraste, entra en log10 y solo para segmentar. El objetivo del modelado se mantiene en quetzales.

## Procedimiento

Pipeline `VectorAssembler → StandardScaler (media 0, desviación 1) → KMeans`, con semilla fija (`config.SEMILLA`), `maxIter = 50` e inicialización `k-means||`. Se ajusta con todos los registros de 2025, sin ponderar, para K = 2, 3, 4 y 5.

## Criterio de elección de K

Se usa el **codo del WSSSE, confirmado con el silhouette**:

1. Se descartan los K cuyo cluster más pequeño tiene menos del 5 % de los registros.
2. Se descarta K = 2, porque solo parte los datos en dos mitades.
3. Entre los K restantes se elige el que tiene la mayor reducción del WSSSE frente a K − 1.
4. Se comprueba que su silhouette sea razonable (mayor que 0.25).

| K | Silhouette | Reducción del WSSSE | Cluster más pequeño |
|---|---|---|---|
| 2 | 0.55 | — | 27.5 % |
| 3 | 0.44 | 20.9 % | 13.5 % |
| **4** | **0.48** | **23.7 %** | **13.0 %** |
| 5 | 0.49 | 11.8 % | 4.9 % |

K = 2 tiene el mayor silhouette, pero solo separa a los mayores con mucha antigüedad del resto. El codo está en **K = 4**: el cuarto cluster todavía reduce el WSSSE 23.7 % y el quinto solo 11.8 %. Además, K = 5 deja un cluster de menos del 5 %.

## Resultados (2025, 53,025 registros)

| Perfil | Registros | Edad | Antigüedad | Horas | Salario mediano (P25–P75) |
|---|---|---|---|---|---|
| Jóvenes al inicio de su trayectoria | 46.2 % | 26 | 1.2 | 44 | Q3,000 (Q1,600–Q3,900) |
| Adultos mayores con poca antigüedad | 24.0 % | 46 | 3.0 | 42 | Q3,000 (Q1,500–Q4,000) |
| Jornada extendida | 16.8 % | 29 | 2.0 | 72 | Q3,000 (Q2,000–Q3,900) |
| Trayectoria larga y estable | 13.0 % | 48 | 20.0 | 40 | Q3,800 (Q2,250–Q6,000) |

Edad, antigüedad y horas son medianas.

- Tres de los cuatro perfiles tienen el mismo salario mediano. Solo el de trayectoria larga gana más, y concentra el empleo de gobierno (31 % frente a 12 % en toda la muestra).
- El perfil de jornada extendida gana lo mismo que los demás con unas 30 horas más por semana.
- **Con salario** (K = 3, silhouette 0.47) la separación no mejora. Aparece un grupo de jornada corta y salario bajo (17.7 %, salario mediano de Q960, con mayoría de jornaleros y servicio doméstico) que se define por el propio salario. Por eso se mantiene la segmentación sin salario.

## Descripción de los clusters

Cada cluster se nombra a partir de su centroide estandarizado. Un rasgo ("jóvenes", "mayores", "antigüedad alta", "poca antigüedad", "jornada larga", "jornada corta") se menciona cuando el centroide se aleja al menos media desviación estándar del promedio.

Además del nombre, cada perfil se describe con:

- Tamaño.
- Medianas de edad, antigüedad y horas.
- Salario mediano con su P25 y P75.
- Categoría ocupacional, nivel educativo y dominio predominantes.

La visualización usa PCA de dos componentes ajustado con todos los registros y dibuja una muestra de hasta 5,000.

## Alcance

Los perfiles describen los registros analizados de 2025; no son estimaciones poblacionales (para ello se usaría `FACTOR`). Las diferencias de salario entre perfiles son asociaciones, no efectos causales. La etiqueta de cluster no se usa como predictor en los modelos supervisados.
