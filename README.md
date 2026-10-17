# Perfiles de carrera y detección de anomalías — Big Data Derby

Segundo modelo de machine learning para el pipeline **Big Data Derby** (carreras de caballos de la NYRA en Aqueduct, Belmont y Saratoga, 2019), construido en **Databricks + Unity Catalog**. Es un modelo **no supervisado y post-carrera**: agrupa cada caballo-carrera según **cómo corrió** (la forma de su curva de velocidad GPS) y marca las carreras **atípicas** (posibles incidentes o fallas del tracking).

> Curso de Big Data — Maestría en Ciencia de Datos, Universidad Pontificia Bolivariana.
> Autor: **Sebastián Muñoz Palacio**

**Stack:** Databricks (serverless, Photon) · Unity Catalog · Delta Lake · PySpark · Spark MLlib (`KMeans`) · MLflow

---

## Resultados principales

| Resultado | Valor |
|---|---|
| Caballo-carreras en la tabla Gold nueva | 14,915 (14,907 usables para el modelo) |
| Saltos de GPS detectados y excluidos (frames > 90 km/h) | 0.03 % de los frames; el máximo crudo era **3,502 km/h** |
| `k` elegido | **3** (silhouette = 0.557) |
| Estilo con más top 3 | **Cerrador**: 46.6 % top 3 y 16.6 % ganadores (promedio general: 40.3 % y 13.4 %) |
| Estilo con menos top 3 | **Puntero que se apaga**: 25.7 % top 3 |
| Anomalías marcadas (top 2 % de distancia al centroide) | 299 caballo-carreras |
| De las 5 anomalías más fuertes | 3 son fallas del tracking y 2 son posibles incidentes |

---

## El problema

El módulo ya tenía un clasificador (`GBTClassifier`) que predice si un caballo queda en el **top 3 antes de la largada**. Por eso excluye la velocidad GPS: esa información no existe antes de la carrera y usarla sería *data leakage*.

Este modelo responde otra pregunta: **una vez corrida la carrera, ¿cómo corrió cada caballo?** Al ser descriptivo, puede usar el tracking GPS sin leakage. Las preguntas que se buscaba responder:

1. ¿Hay **estilos** de carrera reconocibles (punteros, cerradores, ritmo parejo)?
2. ¿Algún estilo llega **más seguido al top 3**?
3. ¿Qué carreras son **atípicas**, y son incidentes reales o ruido del sensor?

---

## Parte 1 · Tabla Gold `gold_derby_race_dynamics`

**Grano:** una fila por **caballo en una carrera** (`track_id, race_date, race_number, program_number`). La llave `start_id` es el mismo hash `sha2` que usa `gold_derby_ml_features`, así que las dos tablas se pueden cruzar.

La tabla existente solo tenía promedios y máximos de velocidad: describía *qué tan rápido*, pero no *cómo*. La tabla nueva captura la **forma de la carrera en el tiempo**:

| Feature | Qué captura |
|---|---|
| `speed_q1` … `speed_q4` | Velocidad media en cada cuarto de la carrera, relativo al recorrido de cada caballo |
| `early_speed_ratio` | Arranque: `speed_q1` / promedio de los 4 cuartos |
| `late_speed_ratio` | Cierre: `speed_q4` / promedio de los 4 cuartos |
| `time_to_top_speed_frac` | En qué punto de la carrera (0 a 1) alcanza su velocidad máxima |
| `speed_cv` | Regularidad del ritmo: desviación estándar / media |
| `n_decel_events`, `max_decel_kmh` | Frenadas bruscas: caídas de más de 8 km/h entre frames consecutivos |
| `race_duration_s`, `n_frames` | Duración y cantidad de frames válidos |
| `n_gps_spikes`, `n_frame_gaps` | Calidad del tracking: saltos de GPS excluidos y frames faltantes |

### Hallazgo de calidad de datos: saltos de GPS

En la primera corrida había caballos "corriendo" a más de 500 km/h, con un máximo de 3,502 km/h. Son saltos de posición del GPS, no velocidades reales. El umbral se fijó **con la distribución de los datos** y no a ojo:

| Corte | Frames por encima | % |
|---|---|---|
| 75 km/h | 569,514 | 10.9 % |
| 80 km/h | 50,205 | 0.96 % |
| **90 km/h** | **1,588** | **0.03 %** |
| 100 km/h | 808 | — |

Con 75 km/h se perdían velocidades reales de caballos en punta. Con **90 km/h** solo se excluye el 0.03 %, que sí son saltos. Los frames excluidos no se pierden: quedan contados en `n_gps_spikes`.

**Validación antes de escribir:** la llave es única y sin nulos, `start_id` cruza con `gold_derby_ml_features`, y el `describe()` de las velocidades por cuarto da valores plausibles para caballos de carrera.

---

## Parte 2 · Modelo: clustering + detección de anomalías

### 2.1 Estilos de carrera con `KMeans`

- **Features:** solo las de **forma** (`early_speed_ratio`, `late_speed_ratio`, `time_to_top_speed_frac`, `speed_cv`). Las velocidades absolutas mezclan *qué tan rápido* con *cómo*.
- **Pipeline:** `VectorAssembler` → `StandardScaler` → `KMeans` (semilla 42).
- **Filtro:** se excluyen los caballo-carreras con menos de 40 frames (unos 10 s de tracking).
- **Elección de `k`:** se usaron el método del codo (WSSSE) y la silhouette para k = 2…8. Con k = 2 la silhouette es mayor (0.602), pero solo separa "se apaga / no se apaga". Con **k = 3** aparece el cerrador como grupo propio. De k = 4 en adelante la silhouette baja y los grupos son difíciles de interpretar.

| Cluster | Estilo | Caballo-carreras | Cómo corre | % top 3 | % ganador | Cuota mediana |
|---|---|---|---|---|---|---|
| 2 | **Cerrador** | 2,431 (16 %) | Sale lento y llega a su velocidad máxima tarde (74 % de la carrera) | **46.6** | 16.6 | 6.5 |
| 0 | **Ritmo parejo** | 9,204 (62 %) | Velocidad estable y pequeña caída al final | 43.8 | 14.5 | 7.2 |
| 1 | **Puntero que se apaga** | 3,272 (22 %) | Sale fuerte (early 1.07) y se cae en el último cuarto (late 0.81) | **25.7** | 8.0 | 8.3 |

**Lectura:** gastar la energía al inicio se paga al final. Los cerradores son los que más llegan al top 3 y los que se apagan rinden 15 puntos por debajo de la media. Es una asociación, no causalidad: un caballo que va perdiendo también "se apaga" porque el jinete deja de exigirlo.

> En las tablas del notebook el cluster 1 aparece como "Errático", que es el nombre que asigna automáticamente la regla del código. La interpretación lo renombra "Puntero que se apaga": su alta variabilidad viene de la caída final, no de un ritmo irregular.

### 2.2 Detección de anomalías

- **Score:** distancia euclidiana, en el espacio estandarizado, de cada caballo-carrera al centroide de **su** cluster.
- **Umbral:** percentil 98 (score > 2.657), que marca **299 anomalías** (2.01 %).
- **Diferencia con las normales:** las anómalas tienen 3.1 frenadas bruscas en promedio (contra 0.95) y el doble de variabilidad. Su velocidad máxima ya es normal (80 km/h), señal de que el filtro de GPS funcionó.

Inspección manual de las 5 más fuertes:

| # | Caballo-carrera | Velocidad por cuarto (km/h) | Llegada | Lectura |
|---|---|---|---|---|
| 1 | SAR 2019-08-04 · R6 · #10 | 61 → 67 → 9 → 0 | 11.º | Ruido de tracking: 6 saltos de GPS y señal perdida |
| 2 | BEL 2019-06-20 · R2 · #2 | 68 → 62 → 11 → 2 | 4.º | Error de tracking: llegó 4.º pero "iba" a 2 km/h |
| 3 | BEL 2019-05-17 · R2 · #4 | 10 → 38 → 54 → 62 | 5.º | Tracking corto: solo 42 frames |
| 4 | BEL 2019-05-24 · R7 · #2 | 70 → 74 → 51 → 1 | 9.º | Posible incidente; hay que confirmarlo con el reporte de la carrera |
| 5 | AQU 2019-01-12 · R7 · #5 | 63 → 72 → 61 → 2 | 9.º | Mismo patrón que el 4 |

**Conclusión:** el detector funciona, pero las anomalías más extremas son de **calidad de datos**. Los incidentes reales están probablemente entre los 100 casos marcados como "Posible incidente", con scores más moderados. Como mejora se propone una regla adicional: un caballo que figura en la llegada pero promedia menos de 15 km/h en el último cuarto tiene el tracking truncado o congelado.

### 2.3 MLflow y tabla de salida

- **MLflow:** el run se registra en un experimento con nombre fijo (`/Users/<usuario>/09-derby/derby_race_profiles`) con:
  - parámetros: `k`, features, umbral de anomalía, umbrales de Gold, semilla;
  - métricas: silhouette, WSSSE, % de anomalías;
  - las tablas de elección de `k` y de perfiles de cluster (CSV);
  - el `PipelineModel` completo con firma.
- **Tabla de salida:** `gold_derby_race_profiles` (14,907 filas) guarda, por caballo-carrera, `start_id`, cluster, nombre del cluster, `anomaly_score`, `is_anomaly` y el `mlflow_run_id` que la generó.

---

## Respuestas a las preguntas

**1. ¿Por qué este modelo sí puede usar la velocidad GPS si el clasificador top-3 la prohíbe?**
Por el *momento de uso*. El clasificador predice **antes** de la largada, cuando el GPS de esa carrera no existe; entrenarlo con él le daría información del futuro. Este modelo se usa **después** de la llegada, cuando el tracking completo ya existe. Sí sería leakage usar estos clusters como feature del clasificador pre-carrera de esa misma carrera.

**2. ¿Por qué estandarizar antes de `KMeans`?**
`KMeans` agrupa por distancia euclidiana, así que cada feature pesa según su escala. Las velocidades en km/h varían en decenas y los ratios en centésimas. Sin escalar, las velocidades dominarían y los clusters separarían caballos rápidos de lentos, no estilos.

**3. ¿Por qué ratios de forma y no velocidades absolutas?**
La velocidad absoluta depende de la pista, la superficie, la distancia y la calidad del caballo. Los ratios comparan a cada caballo **contra sí mismo**, así que capturan la forma de correr. Con velocidades absolutas, los clusters terminarían separando pistas o distancias.

**4. ¿Una anomalía es necesariamente un incidente?**
No. La distancia al centroide solo dice que el perfil es raro, no por qué. Otras causas son los saltos de GPS, los frames faltantes, un tracking corto o un tracking que sigue después de la meta. Para distinguirlas se usan las columnas de calidad (`n_gps_spikes`, `n_frame_gaps`, `n_frames`) y la posición de llegada. En estos datos, 3 de las 5 anomalías más fuertes eran fallas del sensor.

---

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| [`config.py`](config.py) | Configuración del pipeline Derby con las constantes nuevas `T_GOLD_RACE_DYNAMICS` y `T_GOLD_RACE_PROFILES` |
| [`notebooks/03-gold/05_gold_race_dynamics.py`](notebooks/03-gold/05_gold_race_dynamics.py) | Parte 1: tabla Gold nueva, validación y verificación |
| [`notebooks/04-ml/04_cluster_race_profiles.py`](notebooks/04-ml/04_cluster_race_profiles.py) | Parte 2: clustering, elección de `k`, anomalías, MLflow y respuestas |
| [`notebooks_html/`](notebooks_html/) | Los mismos notebooks **con sus resultados**. Descárgalos y ábrelos en el navegador |

---

## Cómo reproducirlo

1. Importar los notebooks en `09-derby/03-gold/` y `09-derby/04-ml/` del workspace. Deben estar ya corridos los notebooks de Bronze, Silver y Gold del módulo.
2. Agregar a `00-setup/config` las dos constantes nuevas:
   ```python
   T_GOLD_RACE_DYNAMICS = f"{TABLE_PREFIX}.gold_derby_race_dynamics"
   T_GOLD_RACE_PROFILES = f"{TABLE_PREFIX}.gold_derby_race_profiles"
   ```
3. Correr `05_gold_race_dynamics` y después `04_cluster_race_profiles`, en serverless. El entrenamiento toma segundos.
