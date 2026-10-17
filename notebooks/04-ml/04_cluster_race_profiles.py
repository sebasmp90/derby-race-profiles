# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Big Data Derby — ML · Perfiles de carrera y detección de anomalías
# MAGIC
# MAGIC ## Propuesta
# MAGIC
# MAGIC Un **segundo modelo**, no supervisado, sobre `gold_derby_race_dynamics`:
# MAGIC
# MAGIC 1. **Clustering de estilos de carrera** con `KMeans`: agrupa cada caballo-carrera por la **forma** de su curva
# MAGIC    de velocidad (cómo arranca, cuándo alcanza la punta, cómo cierra, qué tan regular es).
# MAGIC 2. **Detección de anomalías**: la distancia de cada caballo-carrera al centroide de su cluster es el score; el
# MAGIC    2 % más lejano se marca como anómalo (posible tropiezo, estorbo, caballo descolgado o tracking defectuoso).
# MAGIC
# MAGIC Es distinto del clasificador top-3 porque no predice un resultado antes de la largada: **describe** una carrera
# MAGIC ya corrida. Por eso puede usar el tracking GPS sin *data leakage* (ver la propuesta completa en
# MAGIC `03-gold/05_gold_race_dynamics`).

# COMMAND ----------

# MAGIC %run ../00-setup/config

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature

mlflow.set_registry_uri("databricks-uc")

T_GOLD_RACE_PROFILES = globals().get("T_GOLD_RACE_PROFILES", f"{TABLE_PREFIX}.gold_derby_race_profiles")

SHAPE_FEATURES = ["early_speed_ratio", "late_speed_ratio", "time_to_top_speed_frac", "speed_cv"]
MIN_FRAMES = 40          # ~10 s de tracking a 4 Hz: menos que eso no tiene un perfil completo
K_RANGE = range(2, 9)
ANOMALY_PCT = 0.02       # top 2 % de distancia al centroide
SEED = 42

dynamics = spark.table(T_GOLD_RACE_DYNAMICS)
starts = spark.table(T_SILVER_STARTS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Datos de entrada
# MAGIC
# MAGIC `KMeans` no acepta `NULL` en el vector de features. Se excluyen los caballo-carrera con menos de `MIN_FRAMES`
# MAGIC frames válidos (no alcanzan a tener cuatro cuartos con sentido) o con alguna feature de forma nula. Se reporta
# MAGIC cuántos quedan fuera.
# MAGIC
# MAGIC **Features de forma** (no velocidades absolutas): `early_speed_ratio`, `late_speed_ratio`,
# MAGIC `time_to_top_speed_frac`, `speed_cv`. Ver la pregunta 3 al final.

# COMMAND ----------

usable_filter = F.col("n_frames") >= MIN_FRAMES
for c in SHAPE_FEATURES:
    usable_filter = usable_filter & F.col(c).isNotNull() & ~F.isnan(c)

data = dynamics.filter(usable_filter)
n_total, n_data = dynamics.count(), data.count()
print(f"Caballo-carrera en Gold: {n_total:,} | usables para el modelo: {n_data:,} "
      f"| excluidos: {n_total - n_data:,} ({100 * (n_total - n_data) / n_total:.1f} %)")
display(data.select(*SHAPE_FEATURES).describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Elección de `k`: método del codo + silhouette
# MAGIC
# MAGIC Se ajusta el escalado una sola vez (`VectorAssembler → StandardScaler` con media 0 y desviación 1) y se prueba
# MAGIC `KMeans` con `k = 2..8`. Para cada `k`:
# MAGIC
# MAGIC - **Costo (WSSSE)**: suma de distancias al cuadrado de cada punto a su centroide. Siempre baja al subir `k`; se
# MAGIC   busca el "codo", donde deja de bajar mucho.
# MAGIC - **Silhouette** (`ClusteringEvaluator`): de −1 a 1, qué tan separados están los grupos. Más alto es mejor.

# COMMAND ----------

assembler = VectorAssembler(inputCols=SHAPE_FEATURES, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
prep_model = Pipeline(stages=[assembler, scaler]).fit(data)
scaled = prep_model.transform(data).select("start_id", "features")

# Matriz escalada en numpy (el dataset es pequeño): sirve para calcular el costo de cada k
X = np.array([r["features"].toArray() for r in scaled.select("features").collect()])

evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction", metricName="silhouette")
rows = []
for k in K_RANGE:
    km = KMeans(k=k, seed=SEED, featuresCol="features", predictionCol="prediction").fit(scaled)
    sil = evaluator.evaluate(km.transform(scaled))
    centers = np.array(km.clusterCenters())
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    cost = float(d2.min(axis=1).sum())
    rows.append((k, cost, sil))
    print(f"k={k}: costo={cost:,.1f} | silhouette={sil:.4f}")

k_scores = pd.DataFrame(rows, columns=["k", "costo_wssse", "silhouette"])

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(k_scores["k"], k_scores["costo_wssse"], "o-", color="#1f6feb", label="Costo (WSSSE)")
ax1.set_xlabel("k (número de clusters)")
ax1.set_ylabel("Costo (WSSSE)", color="#1f6feb")
ax2 = ax1.twinx()
ax2.plot(k_scores["k"], k_scores["silhouette"], "s--", color="#d4630b", label="Silhouette")
ax2.set_ylabel("Silhouette", color="#d4630b")
ax1.set_title("Método del codo y silhouette por k")
fig.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regla de elección
# MAGIC
# MAGIC `k = 2` casi siempre tiene la silhouette más alta porque solo separa "normal" de "raro", y eso no describe
# MAGIC estilos. Se elige el `k` con **mayor silhouette entre 3 y 6**, revisando que coincida con el codo del costo:
# MAGIC suficientes grupos para nombrar estilos distintos, pocos para que cada uno sea interpretable.
# MAGIC
# MAGIC Si el gráfico sugiere otro valor, cámbialo en `K_FINAL` y escribe aquí la justificación.

# COMMAND ----------

candidates = k_scores[(k_scores["k"] >= 3) & (k_scores["k"] <= 6)]
K_FINAL = int(candidates.sort_values("silhouette", ascending=False).iloc[0]["k"])
print(f"k elegido: {K_FINAL} (silhouette = {k_scores.set_index('k').loc[K_FINAL, 'silhouette']:.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Modelo final e interpretación de los clusters
# MAGIC
# MAGIC Pipeline completo `VectorAssembler → StandardScaler → KMeans(k = K_FINAL)`. Para interpretar se miran dos
# MAGIC cosas:
# MAGIC
# MAGIC - **Centroides en unidades originales** (promedio de cada feature por cluster): lo que se reporta.
# MAGIC - **Centroides estandarizados** (z-score): cuánto se aleja cada grupo del promedio general. Con ellos se
# MAGIC   propone un nombre automático, que debes revisar.
# MAGIC
# MAGIC Recuerda que todos los caballos salen desde parado, así que `early_speed_ratio` es menor que 1 para casi
# MAGIC todos: lo que distingue a un puntero es tener un arranque **más alto que el resto**, no mayor que 1.

# COMMAND ----------

pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=SHAPE_FEATURES, outputCol="features_raw"),
    StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True),
    KMeans(k=K_FINAL, seed=SEED, featuresCol="features", predictionCol="cluster"),
])
model = pipeline.fit(data)
pred = model.transform(data)
silhouette = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                                 metricName="silhouette").evaluate(pred)
centers_z = np.array(model.stages[-1].clusterCenters())
print(f"Silhouette del modelo final (k={K_FINAL}): {silhouette:.4f}")


def name_cluster(z):
    """Nombre sugerido a partir del centroide estandarizado (orden de SHAPE_FEATURES)."""
    early, late, ttt, cv = z
    if cv > 1.0:
        return "Errático"
    if early > 0.4 and late < -0.4:
        return "Puntero que se apaga"
    if late > 0.4 and early < -0.4:
        return "Cerrador"
    if ttt > 0.6 and late > 0:
        return "Remate tardío"
    if ttt < -0.6:
        return "Punta temprana"
    if max(abs(early), abs(late), abs(cv)) < 0.4:
        return "Ritmo parejo"
    return "Mixto"


names = {}
for i, z in enumerate(centers_z):
    base = name_cluster(z)
    names[i] = base if base not in names.values() else f"{base} ({i})"
names_df = spark.createDataFrame([(i, n) for i, n in names.items()], "cluster INT, cluster_name STRING")

display(spark.createDataFrame(
    [(i, names[i], *[float(round(v, 2)) for v in z]) for i, z in enumerate(centers_z)],
    "cluster INT, nombre_sugerido STRING, " + ", ".join(f"z_{c} DOUBLE" for c in SHAPE_FEATURES)))

# COMMAND ----------

profiles = (pred.join(names_df, "cluster")
                .groupBy("cluster", "cluster_name")
                .agg(F.count("*").alias("caballo_carreras"),
                     *[F.round(F.avg(c), 3).alias(c) for c in SHAPE_FEATURES],
                     *[F.round(F.avg(f"speed_q{q}"), 1).alias(f"speed_q{q}") for q in range(1, 5)])
                .withColumn("pct", F.round(100 * F.col("caballo_carreras") / F.lit(n_data), 1))
                .orderBy("cluster"))
display(profiles)

# Curva de velocidad promedio por cluster: la forma de cada estilo
prof_pd = profiles.toPandas()
fig, ax = plt.subplots(figsize=(8, 4))
for _, r in prof_pd.iterrows():
    ax.plot(["Q1", "Q2", "Q3", "Q4"], [r[f"speed_q{q}"] for q in range(1, 5)], "o-",
            label=f"{int(r['cluster'])} · {r['cluster_name']} ({r['pct']} %)")
ax.set_ylabel("Velocidad media (km/h)")
ax.set_title("Perfil de velocidad promedio por cluster")
ax.legend(fontsize=8)
fig.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ¿Algún estilo gana más seguido?
# MAGIC
# MAGIC Cruce con `silver_derby_starts`. Compara `% top 3` y `% ganador` de cada cluster contra el promedio general
# MAGIC (≈ 3 / tamaño del campo). Como el perfil es **post-carrera**, esto describe cómo ganan los ganadores, no sirve
# MAGIC para predecir: un caballo que va ganando tiende a cerrar fuerte precisamente porque va ganando.

# COMMAND ----------

results = (pred.select("start_id", *START_KEYS, "cluster")
               .join(names_df, "cluster")
               .join(starts.select(*START_KEYS, "position_at_finish", "is_top3", "is_winner", "odds_decimal"),
                     START_KEYS, "left"))

by_cluster = (results.groupBy("cluster", "cluster_name")
                     .agg(F.count("*").alias("caballo_carreras"),
                          F.round(100 * F.avg(F.col("is_top3").cast("int")), 1).alias("pct_top3"),
                          F.round(100 * F.avg(F.col("is_winner").cast("int")), 1).alias("pct_ganador"),
                          F.round(F.avg("position_at_finish"), 2).alias("posicion_media"),
                          F.round(F.expr("percentile_approx(odds_decimal, 0.5)"), 2).alias("cuota_mediana"))
                     .orderBy(F.desc("pct_top3")))
overall = results.agg(F.round(100 * F.avg(F.col("is_top3").cast("int")), 1).alias("pct_top3_general"),
                      F.round(100 * F.avg(F.col("is_winner").cast("int")), 1).alias("pct_ganador_general"))
display(by_cluster)
display(overall)

best = by_cluster.first()
print(f"Estilo con más top 3: cluster {best['cluster']} · {best['cluster_name']} "
      f"({best['pct_top3']} % top 3, {best['pct_ganador']} % ganadores)")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación de los clusters** (k = 3, silhouette = 0.557; 14,907 caballo-carreras)
# MAGIC
# MAGIC | Cluster | Nombre | n (%) | Cómo corre | % top 3 | % ganador | Posición media | Cuota mediana |
# MAGIC |---|---|---|---|---|---|---|---|
# MAGIC | 0 | Ritmo parejo | 9,204 (62 %) | Arranca al ritmo de su media (early 1.00), pierde poco al final (late 0.92), llega a su velocidad máxima temprano (20 % de la carrera) y varía poco (CV 0.13). Es el estilo "por defecto". | 43.8 | 14.5 | 4.26 | 7.2 |
# MAGIC | 1 | Puntero que se apaga | 3,272 (22 %) | Sale más rápido que su media (early 1.07) y se cae fuerte en el último cuarto (late 0.81); por eso tiene la mayor variabilidad (CV 0.18). El nombre automático fue "Errático", pero la variabilidad viene de la caída, no de un ritmo irregular. | 25.7 | 8.0 | 5.39 | 8.3 |
# MAGIC | 2 | Cerrador | 2,431 (16 %) | Sale por debajo de su media (early 0.96), sostiene la velocidad al final (late 0.99) y alcanza su máximo tarde (74 % de la carrera). Es el más constante (CV 0.11). | 46.6 | 16.6 | 4.12 | 6.5 |
# MAGIC
# MAGIC Referencia general: 40.3 % top 3 y 13.4 % ganadores.
# MAGIC
# MAGIC **Lectura:** lo que separa los estilos es cómo se reparte la energía. Los **cerradores** son los que más
# MAGIC terminan en el top 3 (46.6 %, +6.3 pp sobre la media) y los que más ganan; el **ritmo parejo** queda apenas por encima
# MAGIC de la media. El grupo que **sale fuerte y se apaga** rinde claramente peor (25.7 % top 3, −14.6 pp): gastar la
# MAGIC energía al inicio se paga en el último cuarto, que es donde se define la carrera. El mercado ya lo descuenta en parte
# MAGIC (cuota mediana 6.5 para cerradores vs 8.3 para los que se apagan), pero la diferencia en top 3 es más grande que la de las cuotas.
# MAGIC
# MAGIC Nota: la silhouette con k = 2 fue mayor (0.602), pero solo separaba "se apaga / no se apaga"; con k = 3 aparece el
# MAGIC cerrador como grupo propio, que es justo el que más interesa para la pregunta de negocio.
# MAGIC
# MAGIC Precaución: es una asociación, no causalidad. Un caballo que va perdiendo puede "apagarse" porque el jinete ya no lo exige;
# MAGIC el estilo es en parte consecuencia del resultado, no solo causa.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Detección de carreras anómalas
# MAGIC
# MAGIC **Score de anomalía** = distancia euclidiana (en el espacio estandarizado) de cada caballo-carrera al centroide
# MAGIC de **su** cluster. Un caballo-carrera lejos de todos los estilos conocidos no se parece a ninguna forma
# MAGIC "normal" de correr.
# MAGIC
# MAGIC **Umbral:** percentil 98 del score → el 2 % más lejano queda con `is_anomaly = true`.

# COMMAND ----------

pred_pd = pred.select("start_id", "cluster", "features").toPandas()
Xf = np.array([v.toArray() for v in pred_pd["features"]])
pred_pd["anomaly_score"] = np.linalg.norm(Xf - centers_z[pred_pd["cluster"].to_numpy()], axis=1)
ANOMALY_THRESHOLD = float(np.quantile(pred_pd["anomaly_score"], 1 - ANOMALY_PCT))
pred_pd["is_anomaly"] = pred_pd["anomaly_score"] >= ANOMALY_THRESHOLD

scores = spark.createDataFrame(pred_pd[["start_id", "anomaly_score", "is_anomaly"]])
pct_anom = 100 * pred_pd["is_anomaly"].mean()
print(f"Umbral (percentil {100 * (1 - ANOMALY_PCT):.0f}): {ANOMALY_THRESHOLD:.3f} "
      f"| anómalos: {int(pred_pd['is_anomaly'].sum()):,} ({pct_anom:.2f} %)")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(pred_pd["anomaly_score"], bins=80, color="#1f6feb")
ax.axvline(ANOMALY_THRESHOLD, color="#d4630b", linestyle="--", label=f"Umbral p{100 * (1 - ANOMALY_PCT):.0f}")
ax.set_xlabel("Distancia al centroide (score de anomalía)")
ax.set_ylabel("Caballo-carreras")
ax.legend()
fig.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Anómalos vs. normales
# MAGIC
# MAGIC Si el score captura incidentes, los anómalos deberían tener más `n_decel_events`, `max_decel_kmh` más negativo,
# MAGIC más `speed_cv` y peor posición. Si en cambio tienen muchos `n_gps_spikes` (saltos excluidos en Gold), huecos
# MAGIC (`n_frame_gaps`) o pocos `n_frames`, el score está capturando **ruido de tracking**.

# COMMAND ----------

detail = (results.join(scores, "start_id")
                 .join(dynamics.select("start_id", "n_frames", "race_duration_s", "n_decel_events", "max_decel_kmh",
                                       "speed_cv", "max_speed_kmh", "raw_max_speed_kmh", "n_gps_spikes", "n_frame_gaps",
                                       *SHAPE_FEATURES[:3],
                                       *[f"speed_q{q}" for q in range(1, 5)]),
                       "start_id"))

display(detail.groupBy("is_anomaly").agg(
    F.count("*").alias("caballo_carreras"),
    F.round(F.avg("n_decel_events"), 2).alias("decel_events_prom"),
    F.round(F.avg("max_decel_kmh"), 2).alias("max_decel_prom"),
    F.round(F.avg("speed_cv"), 3).alias("speed_cv_prom"),
    F.round(F.avg("n_gps_spikes"), 2).alias("saltos_gps_prom"),
    F.round(F.avg("n_frame_gaps"), 2).alias("huecos_prom"),
    F.round(F.avg("max_speed_kmh"), 1).alias("vel_max_prom"),
    F.round(F.avg("n_frames"), 0).alias("frames_prom"),
    F.round(F.avg("position_at_finish"), 2).alias("posicion_prom"),
).orderBy("is_anomaly"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspección manual: las 5 anomalías más fuertes
# MAGIC
# MAGIC `diagnostico_sugerido` aplica reglas simples para orientar la revisión (no reemplaza tu lectura):
# MAGIC
# MAGIC - **Tracking ruidoso:** 3 o más saltos de GPS excluidos en Gold (`n_gps_spikes`). Aunque los saltos ya no
# MAGIC   entran a las features, un tracking con tantos saltos tiene más ruido residual en los frames que quedan.
# MAGIC - **Frames faltantes:** más del 2 % de los frames con huecos (`n_frame_gaps / n_frames`).
# MAGIC - **Tracking corto:** muy pocos frames o duración anormalmente baja.
# MAGIC - **Posible incidente:** desaceleraciones bruscas sin señales de ruido.
# MAGIC - **Descolgado / se detuvo:** velocidad del último cuarto muy por debajo del resto.

# COMMAND ----------

diag = (detail.filter("is_anomaly")
              .withColumn("diagnostico_sugerido",
                          F.when(F.col("n_gps_spikes") >= 3, "Tracking ruidoso (varios saltos de GPS)")
                           .when(F.col("n_frame_gaps") > 0.02 * F.col("n_frames"), "Frames faltantes")
                           .when(F.col("n_frames") < 2 * MIN_FRAMES, "Tracking corto")
                           .when(F.col("n_decel_events") >= 2, "Posible incidente (frenadas bruscas)")
                           .when(F.col("late_speed_ratio") < 0.85, "Descolgado / se detuvo")
                           .otherwise("Perfil atípico sin causa evidente"))
              .orderBy(F.desc("anomaly_score")))
display(diag.groupBy("diagnostico_sugerido").count().orderBy(F.desc("count")))
display(diag.select("start_id", *START_KEYS, "cluster_name", F.round("anomaly_score", 2).alias("score"),
                    "diagnostico_sugerido", "n_decel_events", F.round("max_decel_kmh", 1).alias("max_decel"),
                    "n_gps_spikes", "n_frame_gaps", F.round("max_speed_kmh", 1).alias("vel_max"),
                    *[F.round(f"speed_q{q}", 1).alias(f"q{q}") for q in range(1, 5)],
                    "n_frames", "position_at_finish", "odds_decimal").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC **Resumen de las 299 anomalías** (umbral p98 = 2.657):
# MAGIC
# MAGIC | | Normales (14,608) | Anómalas (299) |
# MAGIC |---|---|---|
# MAGIC | Frenadas bruscas promedio | 0.95 | 3.13 |
# MAGIC | Peor frenada promedio (km/h por frame) | −8.2 | −17.5 |
# MAGIC | Coeficiente de variación | 0.134 | 0.287 |
# MAGIC | Saltos de GPS excluidos promedio | 0.09 | 0.75 |
# MAGIC | Velocidad máxima promedio (km/h) | 79.6 | 80.4 |
# MAGIC
# MAGIC Con el filtro de 90 km/h, la velocidad máxima de las anomalías ya es igual a la de los normales (en la primera
# MAGIC corrida, sin filtro, era 272 km/h). Lo que ahora las separa es la **forma**: más frenadas y mucha más variabilidad.
# MAGIC
# MAGIC Diagnóstico sugerido: Descolgado / se detuvo 140 · Posible incidente 100 · Perfil atípico 27 · Tracking ruidoso 26 · Tracking corto 6.
# MAGIC
# MAGIC **Lectura de las 5 anomalías más fuertes** (velocidades promedio por cuarto, en km/h):
# MAGIC
# MAGIC | # | Caballo-carrera | Score | q1 → q4 | Frames | Llegada (cuota) | Lectura |
# MAGIC |---|---|---|---|---|---|---|
# MAGIC | 1 | SAR 2019-08-04 · R6 · #10 | 26.45 | 61 → 67 → 9 → 0 | 215 | 11.º (34.25) | **Ruido de tracking.** 6 saltos de GPS excluidos y la velocidad cae a 0 en la segunda mitad. Corrió mal (11.º, era muy poco favorito), pero un caballo que termina la carrera no promedia 0 km/h en el último cuarto: la señal se perdió. |
# MAGIC | 2 | BEL 2019-06-20 · R2 · #2 | 25.26 | 68 → 62 → 11 → 2 | 304 | 4.º (9.5) | **Error de tracking.** Llegó 4.º, así que no se detuvo, pero el tracking dice que en la segunda mitad iba a 2–11 km/h. Sin frenadas bruscas (peor −4.3): la velocidad se apaga gradualmente, típico de posiciones congeladas o de tracking desfasado. |
# MAGIC | 3 | BEL 2019-05-17 · R2 · #4 | 18.14 | 10 → 38 → 54 → 62 | 42 | 5.º (8.7) | **Tracking corto.** Solo 42 frames (~10 s, contra ~300 normales): el tracking empieza tarde o está incompleto. El perfil "cerrador" es un artefacto de que el primer cuarto arranca casi parado. Pasó el filtro de `MIN_FRAMES = 40` por muy poco. |
# MAGIC | 4 | BEL 2019-05-24 · R7 · #2 | 17.20 | 70 → 74 → 51 → 1 | 322 | 9.º (15.9) | **Candidato a incidente real, pero dudoso.** Fue rápido la primera mitad y se derrumbó al final; el 9.º puesto es coherente con un caballo que se apagó o fue frenado por el jinete. Sin embargo, 1 km/h en el último cuarto es caminar, y el caballo terminó la carrera: hay que revisarlo con el reporte de la carrera. |
# MAGIC | 5 | AQU 2019-01-12 · R7 · #5 | 15.30 | 63 → 72 → 61 → 2 | 290 | 9.º (18.4) | **Mismo patrón que el 4.** Ritmo normal tres cuartos y caída a ~2 km/h al final. Puede ser un caballo que se detuvo, pero también tracking que siguió corriendo después de la meta o se congeló. |
# MAGIC
# MAGIC **Conclusión:** de las 5 anomalías más fuertes, **3 son claramente problemas de datos** (1, 2 y 3) y **2 son candidatas a
# MAGIC incidente** (4 y 5) que no se pueden confirmar solo con el tracking. El detector funciona, pero lo primero que encuentra
# MAGIC son fallas del sensor: las anomalías más extremas son de **calidad de datos**, y los incidentes reales probablemente
# MAGIC están en el grupo "Posible incidente" (100 casos), con scores más moderados.
# MAGIC
# MAGIC Mejora propuesta: agregar una regla `speed_q4 < 15 km/h` y `position_at_finish` no nula → "Tracking truncado o congelado".
# MAGIC Un caballo que figura en la llegada no puede promediar velocidad de caminata en el último cuarto. Con esa regla, varios casos que hoy caen
# MAGIC en "Descolgado / se detuvo" (140) pasarían a ruido de tracking.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Registro en MLflow y tabla de salida
# MAGIC
# MAGIC Se loguean los parámetros, las métricas, la tabla de `k`, los gráficos y el `PipelineModel` completo.
# MAGIC `dfs_tmpdir=MLFLOW_DFS_TMPDIR` es necesario en serverless: Spark ML se guarda pasando por un Volume de UC.
# MAGIC
# MAGIC Se fija un experimento con nombre explícito: así el run no depende del experimento automático del notebook
# MAGIC (si el notebook se borra o se reimporta, ese experimento queda eliminado y `start_run` falla con
# MAGIC `RESOURCE_DOES_NOT_EXIST ... lifecycle stage is: deleted`).

# COMMAND ----------

_user = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT_PATH = f"/Users/{_user}/09-derby/derby_race_profiles"
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"Experimento MLflow: {EXPERIMENT_PATH}")

# COMMAND ----------

with mlflow.start_run(run_name=f"{USER_INITIALS}_kmeans_derby_race_profiles") as run:
    mlflow.log_params({
        "student": USER_INITIALS,
        "algorithm": "KMeans + distancia al centroide",
        "k": K_FINAL,
        "features": ",".join(SHAPE_FEATURES),
        "scaler": "StandardScaler(withMean=True, withStd=True)",
        "min_frames": MIN_FRAMES,
        "anomaly_pct": ANOMALY_PCT,
        "anomaly_threshold": round(ANOMALY_THRESHOLD, 4),
        "seed": SEED,
        "decel_threshold_kmh": -8.0,
        "max_plausible_kmh": 90.0,
    })
    mlflow.log_metrics({
        "silhouette": silhouette,
        "wssse": float(k_scores.set_index("k").loc[K_FINAL, "costo_wssse"]),
        "pct_anomalies": pct_anom,
        "n_train": n_data,
    })
    for k, cost, sil in rows:
        mlflow.log_metric("silhouette_by_k", sil, step=k)
        mlflow.log_metric("wssse_by_k", cost, step=k)
    mlflow.log_text(k_scores.to_csv(index=False), "k_selection.csv")
    mlflow.log_text(prof_pd.to_csv(index=False), "cluster_profiles.csv")
    sample_input = data.select(*SHAPE_FEATURES).limit(5).toPandas()
    sample_output = model.transform(data.limit(5)).select("cluster").toPandas()
    mlflow.spark.log_model(
        model,
        artifact_path="model",
        signature=infer_signature(sample_input, sample_output),
        dfs_tmpdir=MLFLOW_DFS_TMPDIR,
    )
    RUN_ID = run.info.run_id
print(f"✓ MLflow run: {RUN_ID}")

# COMMAND ----------

output = (pred.select("start_id", *START_KEYS, "cluster")
              .join(names_df, "cluster")
              .join(scores, "start_id")
              .withColumn("anomaly_threshold", F.lit(ANOMALY_THRESHOLD))
              .withColumn("k", F.lit(K_FINAL))
              .withColumn("mlflow_run_id", F.lit(RUN_ID))
              .withColumn("_created_at", F.current_timestamp())
              .select("start_id", *START_KEYS, "cluster", "cluster_name", "anomaly_score", "is_anomaly",
                      "anomaly_threshold", "k", "mlflow_run_id", "_created_at"))

n_out, n_out_ids = output.count(), output.select(F.countDistinct("start_id")).first()[0]
assert n_out == n_out_ids == n_data, f"Salida inconsistente: {n_out} filas, {n_out_ids} ids, {n_data} esperados"

(output.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
       .saveAsTable(T_GOLD_RACE_PROFILES))
print(f"✓ {T_GOLD_RACE_PROFILES}: {n_out:,} filas")
display(spark.table(T_GOLD_RACE_PROFILES).orderBy(F.desc("anomaly_score")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preguntas
# MAGIC
# MAGIC ### 1. ¿Por qué este modelo sí puede usar las features de velocidad GPS, cuando el clasificador top-3 las prohíbe? ¿Qué cambia en el "momento de la predicción"?
# MAGIC
# MAGIC El *leakage* depende de **cuándo** se usa el modelo, no de la columna en sí. El clasificador top-3 se usaría
# MAGIC **antes de la largada**, y en ese instante la velocidad GPS de esa carrera no existe. Entrenarlo con ella le
# MAGIC daría información del futuro que no tendría en producción: métricas excelentes en el notebook e inútiles en la
# MAGIC realidad. Este modelo se usa **después de la llegada**, para describir lo que pasó. En ese momento el tracking
# MAGIC completo ya existe, así que usarlo es legítimo. Lo que sí sería leakage aquí es usar sus clusters como feature
# MAGIC del clasificador pre-carrera de esa misma carrera.
# MAGIC
# MAGIC ### 2. ¿Por qué estandarizar antes de `KMeans`? ¿Qué pasaría con `speed_q1..q4` en km/h junto a ratios entre 0 y 2?
# MAGIC
# MAGIC `KMeans` agrupa por **distancia euclidiana**, así que cada feature pesa según su escala. Las velocidades en km/h
# MAGIC varían en decenas de unidades (45–65), mientras que los ratios varían en centésimas (0,9–1,1). Sin escalar, la
# MAGIC distancia quedaría dominada casi por completo por las velocidades y los ratios no aportarían nada: los clusters
# MAGIC separarían caballos **rápidos de lentos** (o pistas y distancias distintas), no **estilos**. `StandardScaler`
# MAGIC lleva cada feature a media 0 y desviación 1, para que todas aporten en la misma proporción.
# MAGIC
# MAGIC ### 3. ¿Por qué usar ratios de forma en vez de velocidades absolutas para agrupar estilos?
# MAGIC
# MAGIC La velocidad absoluta mezcla **qué tan rápido** con **cómo**. Depende de la pista (tierra o césped), su estado,
# MAGIC la distancia y la calidad del caballo. Un cerrador en una carrera lenta y uno en una rápida tienen velocidades
# MAGIC muy distintas, pero la misma forma: aceleran al final respecto a su propio promedio. Los ratios (`early` y
# MAGIC `late_speed_ratio`), la fracción hasta la punta y el coeficiente de variación están **normalizados por el propio
# MAGIC caballo**, así que comparan formas y no niveles. Si se usaran velocidades absolutas, los clusters terminarían
# MAGIC separando pistas o distancias.
# MAGIC
# MAGIC ### 4. ¿Una anomalía es necesariamente un incidente de carrera? ¿Qué otras causas la disparan y cómo distinguirlas?
# MAGIC
# MAGIC No. La distancia al centroide solo dice que el perfil es **raro**, no por qué. Otras causas:
# MAGIC
# MAGIC | Causa | Señal en las features | Cómo distinguirla |
# MAGIC |---|---|---|
# MAGIC | Salto de GPS | `raw_max_speed_kmh` > 90 (hasta 3.502 km/h en estos datos), `n_gps_spikes` alto | Revisar `step_distance_m` y coordenadas de ese frame; una caída real dura varios frames. Por eso Gold excluye los frames > 90 km/h (0,03 %) |
# MAGIC | Frames faltantes | `n_frame_gaps` alto | Comparar `n_frames` con `race_duration_s × TRACKING_HZ` |
# MAGIC | Carrera o tracking corto | `n_frames` y `race_duration_s` bajos | Comparar con la distancia oficial de la carrera |
# MAGIC | Tracking después de la llegada | `speed_q4` muy baja sin frenadas bruscas | El caballo trota tras cruzar la meta; recortar el tracking a la distancia de la carrera |
# MAGIC | Caballo retirado o que se detuvo | `speed_q4` muy baja y peor posición final | Cruzar con `position_at_finish` y notas del comisario si existen |
# MAGIC | **Incidente real** (tropiezo, estorbo) | Varias `n_decel_events` seguidas de recuperación, peor posición de lo que sugería la cuota | Confirmar con video o con los reportes de la carrera |
# MAGIC
# MAGIC En estos datos se ve claramente. De las 5 anomalías más fuertes, 3 son fallas de tracking: saltos de GPS, un caballo
# MAGIC que llegó 4.º pero "iba" a 2 km/h, y un tracking de solo 42 frames. Solo 2 son candidatas a incidente, y aun así
# MAGIC necesitan confirmación externa.
# MAGIC
# MAGIC Por eso la tabla de salida guarda el score y no un veredicto: las anomalías son **candidatas** a revisión
# MAGIC humana, y las reglas de `diagnostico_sugerido` ayudan a separar ruido de tracking de posibles incidentes.