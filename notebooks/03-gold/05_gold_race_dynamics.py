# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Big Data Derby — Gold · `gold_derby_race_dynamics`
# MAGIC
# MAGIC ## Propuesta
# MAGIC
# MAGIC **Modelo propuesto:** un modelo **no supervisado** de *perfiles de carrera*. Agrupa cada caballo-carrera según
# MAGIC **cómo corrió** (la forma de su curva de velocidad a lo largo de la carrera) con `KMeans`, y usa la distancia de
# MAGIC cada caballo-carrera al centroide de su grupo como **score de anomalía**, para marcar carreras atípicas: tropiezos,
# MAGIC estorbos, caballos descolgados o tracking defectuoso.
# MAGIC
# MAGIC **Por qué es distinto del clasificador top-3:**
# MAGIC
# MAGIC | | Clasificador top-3 (existente) | Perfiles de carrera (este) |
# MAGIC |---|---|---|
# MAGIC | Tipo | Supervisado (`GBTClassifier`) | No supervisado (`KMeans` + distancia al centroide) |
# MAGIC | Pregunta | ¿Quedará entre los 3 primeros? | ¿Cómo corrió y fue una carrera normal? |
# MAGIC | Momento | **Pre-carrera**: antes de la largada | **Post-carrera**: después de la llegada |
# MAGIC | Features | Cuota, peso, distancia, pista, tamaño del campo | Forma del perfil de velocidad GPS |
# MAGIC | Etiqueta | `target_top3` | Ninguna |
# MAGIC
# MAGIC **Por qué puede usar el GPS sin *data leakage*:** el leakage ocurre cuando un modelo usa información que no
# MAGIC existe en el momento en que se hace la predicción. El clasificador top-3 predice **antes** de la largada, y en ese
# MAGIC momento la velocidad de la carrera aún no existe: usarla sería hacer trampa. Este modelo es **descriptivo**:
# MAGIC caracteriza una carrera **ya corrida**, y en ese momento el tracking GPS completo sí está disponible. No predice el
# MAGIC resultado ni se usa para apostar; no hay información del futuro filtrándose al pasado.
# MAGIC
# MAGIC Este notebook construye las **features nuevas** que necesita el modelo (Parte 1). El modelo está en
# MAGIC `04-ml/04_cluster_race_profiles`.

# COMMAND ----------

# MAGIC %run ../00-setup/config

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

tracking = spark.table(T_SILVER_TRACKING)
starts = spark.table(T_SILVER_STARTS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parámetros
# MAGIC
# MAGIC **Umbral de desaceleración brusca: −8 km/h entre frames consecutivos.** Con `TRACKING_HZ` ≈ 4, dos frames
# MAGIC consecutivos están separados ~0,25 s. Perder 8 km/h (2,2 m/s) en 0,25 s es una desaceleración de ~8,9 m/s²,
# MAGIC casi **0,9 g**. Un caballo al galope fluctúa unos pocos km/h entre zancadas, y al terminar la carrera frena en
# MAGIC varios segundos, no en un cuarto de segundo. Una caída así solo se explica por un tropiezo, un frenón por
# MAGIC estorbo o un salto de GPS.
# MAGIC
# MAGIC Por eso la diferencia solo se mide entre frames **consecutivos** (`trakus_index` que avanza en 1). Si faltan
# MAGIC frames, la caída se reparte en más tiempo y compararla con el mismo umbral sería engañoso; esos huecos se
# MAGIC cuentan aparte en `n_frame_gaps`.
# MAGIC
# MAGIC **Umbral de salto de GPS: 90 km/h, elegido con la distribución de los datos.** La primera corrida de este
# MAGIC notebook mostró velocidades imposibles: la máxima promedio por caballo era 82,6 km/h y la de los anómalos 272
# MAGIC km/h. Un salto de GPS es un frame ubicado lejos de su posición real: su `step_distance_m` (y la del frame
# MAGIC siguiente, al "volver") es enorme para 0,25 s. Por ejemplo, 35 m en 0,25 s equivalen a 504 km/h.
# MAGIC
# MAGIC Distribución de `speed_kmh` en `silver_derby_tracking` (5.213.515 frames):
# MAGIC
# MAGIC | Corte | Frames por encima | % | Lectura |
# MAGIC |---|---:|---:|---|
# MAGIC | 75 km/h | 569.514 | 10,9 % | Ruido normal del GPS (percentil 90 = 75,3) |
# MAGIC | 80 km/h | 50.205 | 0,96 % | Ruido normal (percentil 99,5 = 81,0) |
# MAGIC | **90 km/h** | **1.588** | **0,03 %** | **Fin del ruido: caída del 97 % entre 80 y 90** |
# MAGIC | 100 km/h | 808 | 0,015 % | Cola plana de saltos de GPS |
# MAGIC | 500 km/h | 66 | 0,001 % | Saltos extremos (máximo: 3.502 km/h) |
# MAGIC
# MAGIC Un corte en 75 km/h (cerca del récord real de un pura sangre, ~70 km/h) habría borrado el 11 % de los frames,
# MAGIC justo los de la parte rápida de la carrera. Esas velocidades no son saltos: son frames reales **inflados por
# MAGIC el ruido normal del GPS** (el zigzag de las posiciones suma distancia; por eso la mediana es 68,5 km/h). La
# MAGIC distribución cae un 97 % entre 80 y 90 km/h y a partir de ahí se vuelve una cola plana: **90 km/h** separa el
# MAGIC ruido normal de los saltos y excluye solo el 0,03 % de los frames.
# MAGIC
# MAGIC Los frames por encima del umbral se **excluyen** del cálculo y se cuentan en `n_gps_spikes`. Como dejan un hueco
# MAGIC en la secuencia, la "frenada" falsa que sigue a un salto tampoco se cuenta como desaceleración. Las features de
# MAGIC **forma** (ratios) no se ven afectadas por la inflación general, porque se dividen por el promedio del propio
# MAGIC caballo.

# COMMAND ----------

DECEL_THRESHOLD_KMH = -8.0   # caída entre frames consecutivos que cuenta como "brusca"
MAX_PLAUSIBLE_KMH = 90.0     # por encima: salto de GPS (ver tabla de distribución), el frame se excluye
print(f"START_KEYS = {START_KEYS} | TRACKING_HZ = {TRACKING_HZ} | umbral = {DECEL_THRESHOLD_KMH} km/h "
      f"| velocidad máxima plausible = {MAX_PLAUSIBLE_KMH} km/h")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 · Frames válidos y posición dentro de la carrera
# MAGIC
# MAGIC Se descartan los frames con `speed_kmh` nulo (el primer frame de cada caballo no tiene frame anterior). Los
# MAGIC huecos reales de tracking (`n_frame_gaps`) se miden **antes** de quitar los saltos de GPS, para no confundir
# MAGIC ambas cosas; después se excluyen los frames por encima de `MAX_PLAUSIBLE_KMH`.
# MAGIC
# MAGIC **Reparto en cuartos: Opción A (`floor`).** Corta por **avance de la carrera** (`trakus_index` relativo al
# MAGIC rango de ese caballo), que es lo que significa "cuarto de carrera". Con `ntile` los cuartos tendrían el mismo
# MAGIC número de frames, y si faltan frames en un tramo, ese cuarto se "estiraría" hacia el siguiente. Con `floor`, un
# MAGIC hueco solo deja ese cuarto con menos muestras, sin mover las fronteras.
# MAGIC
# MAGIC `pos = (trakus_index − min) / (max − min + 1)` y `quarter = least(floor(4·pos), 3)` → 0, 1, 2, 3.

# COMMAND ----------

w_horse = Window.partitionBy(*START_KEYS)
w_order = Window.partitionBy(*START_KEYS).orderBy("trakus_index")

# Todos los frames con velocidad: sirven para contar huecos reales y saltos de GPS
frames_all = (
    tracking
    .filter(F.col("speed_kmh").isNotNull())
    .withColumn("raw_gap", F.col("trakus_index") - F.lag("trakus_index").over(w_order))
    .withColumn("is_gps_spike", F.col("speed_kmh") > MAX_PLAUSIBLE_KMH)
)
quality = frames_all.groupBy(*START_KEYS).agg(
    F.sum(F.when(F.col("raw_gap") > 1, 1).otherwise(0)).alias("n_frame_gaps"),
    F.sum(F.col("is_gps_spike").cast("int")).alias("n_gps_spikes"),
    F.max("speed_kmh").alias("raw_max_speed_kmh"),
)

tot = frames_all.agg(F.count("*").alias("n"), F.sum(F.col("is_gps_spike").cast("int")).alias("spikes")).first()
print(f"Frames con velocidad: {tot['n']:,} | saltos de GPS (> {MAX_PLAUSIBLE_KMH:.0f} km/h): {tot['spikes']:,} "
      f"({100 * tot['spikes'] / tot['n']:.2f} %)")

frames = (
    frames_all
    .filter(~F.col("is_gps_spike"))
    .withColumn("min_idx", F.min("trakus_index").over(w_horse))
    .withColumn("max_idx", F.max("trakus_index").over(w_horse))
    .withColumn("pos", (F.col("trakus_index") - F.col("min_idx"))
                       / (F.col("max_idx") - F.col("min_idx") + 1))
    .withColumn("quarter", F.least(F.floor(4 * F.col("pos")), F.lit(3)).cast("int"))
    # Diferencia de velocidad contra el frame limpio anterior (solo si es el consecutivo)
    .withColumn("prev_speed", F.lag("speed_kmh").over(w_order))
    .withColumn("prev_idx", F.lag("trakus_index").over(w_order))
    .withColumn("idx_gap", F.col("trakus_index") - F.col("prev_idx"))
    .withColumn("speed_delta",
                F.when(F.col("idx_gap") == 1, F.col("speed_kmh") - F.col("prev_speed")))
)

# Verificación del reparto con un caballo de ejemplo: los 4 cuartos y sus fronteras
sample_key = frames.select(*START_KEYS).limit(1)
display(frames.join(sample_key, START_KEYS)
        .groupBy("quarter")
        .agg(F.count("*").alias("frames"),
             F.min("trakus_index").alias("desde_idx"), F.max("trakus_index").alias("hasta_idx"),
             F.round(F.avg("speed_kmh"), 2).alias("vel_media_kmh"))
        .orderBy("quarter"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2 · Agregación por caballo-carrera
# MAGIC
# MAGIC Todo sale de un solo `groupBy(START_KEYS)`:
# MAGIC
# MAGIC - `speed_q1..q4`: `avg(when(quarter == i, speed_kmh))`. El `when` deja `NULL` fuera del cuarto y `avg` los ignora.
# MAGIC - Frame de velocidad máxima: `max_by(trakus_index, speed_kmh)`.
# MAGIC - Desaceleraciones: `speed_delta` solo existe entre frames consecutivos.
# MAGIC
# MAGIC Además de las features mínimas se agregan columnas de **diagnóstico** (no entran al clustering), para
# MAGIC distinguir incidentes de ruido en la Parte 2: `avg_speed_kmh`, `max_speed_kmh` (sin saltos),
# MAGIC `raw_max_speed_kmh` (con saltos), `n_gps_spikes` y `n_frame_gaps`.

# COMMAND ----------

agg = (
    frames.groupBy(*START_KEYS)
    .agg(
        F.count("*").alias("n_frames"),
        F.min("trakus_index").alias("min_idx"),
        F.max("trakus_index").alias("max_idx"),
        *[F.avg(F.when(F.col("quarter") == q, F.col("speed_kmh"))).alias(f"speed_q{q + 1}") for q in range(4)],
        F.avg("speed_kmh").alias("avg_speed_kmh"),
        F.stddev("speed_kmh").alias("std_speed_kmh"),
        F.max("speed_kmh").alias("max_speed_kmh"),
        F.max_by("trakus_index", "speed_kmh").alias("top_speed_idx"),
        F.sum(F.when(F.col("speed_delta") < DECEL_THRESHOLD_KMH, 1).otherwise(0)).alias("n_decel_events"),
        F.min("speed_delta").alias("max_decel_kmh"),
    )
    .join(quality, START_KEYS, "left")
)

mean_q = (F.col("speed_q1") + F.col("speed_q2") + F.col("speed_q3") + F.col("speed_q4")) / 4
span = F.col("max_idx") - F.col("min_idx")

dynamics = (
    agg
    .withColumn("race_duration_s", span / F.lit(TRACKING_HZ))
    .withColumn("late_speed_ratio", F.col("speed_q4") / mean_q)
    .withColumn("early_speed_ratio", F.col("speed_q1") / mean_q)
    .withColumn("time_to_top_speed_frac",
                F.when(span > 0, (F.col("top_speed_idx") - F.col("min_idx")) / span))
    .withColumn("speed_cv", F.col("std_speed_kmh") / F.col("avg_speed_kmh"))
    .withColumn("max_decel_kmh", F.coalesce("max_decel_kmh", F.lit(0.0)))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3 · Llave `start_id` y columnas finales
# MAGIC
# MAGIC `start_id` es un hash `sha2` de la clave caballo-carrera **con la misma expresión que
# MAGIC `04_gold_ml_features`** (`concat_ws("||", track_id, race_date, race_number, program_number)`), para poder
# MAGIC cruzar ambas tablas. La celda de validación comprueba que de verdad coincide.

# COMMAND ----------

def make_start_id(df):
    """Mismo hash que 03-gold/04_gold_ml_features: sha2 de las START_KEYS unidas con '||'."""
    return df.withColumn(
        "start_id",
        F.sha2(F.concat_ws("||",
                           F.col("track_id"), F.col("race_date").cast("string"),
                           F.col("race_number").cast("string"), F.col("program_number")), 256))

FEATURES = ["n_frames", "race_duration_s", "speed_q1", "speed_q2", "speed_q3", "speed_q4",
            "late_speed_ratio", "early_speed_ratio", "time_to_top_speed_frac", "speed_cv",
            "n_decel_events", "max_decel_kmh"]
DIAGNOSTIC = ["avg_speed_kmh", "max_speed_kmh", "raw_max_speed_kmh", "n_gps_spikes", "n_frame_gaps"]

gold = (
    make_start_id(dynamics)
    .select("start_id", *START_KEYS, *FEATURES, *DIAGNOSTIC)
    .withColumn("_created_at", F.current_timestamp())
)
gold.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4 · Validación pre-escritura
# MAGIC
# MAGIC Falla con `assert` si la llave no es única o tiene nulos. Además compara `start_id` contra
# MAGIC `gold_derby_ml_features` y verifica que todos los caballo-carrera con tracking existan en `silver_derby_starts`.

# COMMAND ----------

n_rows = gold.count()
n_ids = gold.select(F.countDistinct("start_id")).first()[0]
n_null = gold.filter(F.col("start_id").isNull()).count()
print(f"Filas: {n_rows:,} | start_id distintos: {n_ids:,} | start_id nulos: {n_null}")
assert n_rows == n_ids, f"start_id duplicado: {n_rows - n_ids} filas repetidas"
assert n_null == 0, "Hay start_id nulos"

# Todos los caballo-carrera del tracking deben existir en starts
sin_start = gold.join(starts.select(*START_KEYS).distinct(), START_KEYS, "left_anti").count()
print(f"Caballo-carrera con tracking pero sin fila en silver_derby_starts: {sin_start}")

# El hash debe ser el mismo que en gold_derby_ml_features (si la tabla existe)
ml_table = globals().get("T_GOLD_ML_FEATURES")
if ml_table and spark.catalog.tableExists(ml_table):
    ml_ids = spark.table(ml_table).select("start_id").distinct()
    match = gold.join(ml_ids, "start_id").count()
    pct = 100 * match / n_rows
    print(f"start_id que existen en gold_derby_ml_features: {match:,} de {n_rows:,} ({pct:.1f} %)")
    assert pct > 50, ("start_id no coincide con gold_derby_ml_features: copia en make_start_id() la misma "
                      "expresión sha2 que usa el notebook de ml_features")
else:
    print("⚠ No se encontró T_GOLD_ML_FEATURES: no se pudo comparar el hash de start_id.")
print("✓ Llave start_id única y no nula")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plausibilidad de las velocidades por cuarto
# MAGIC
# MAGIC Sin los saltos de GPS, el promedio por cuarto debería quedar entre ~55 y 75 km/h (algo por encima de la
# MAGIC velocidad real por el ruido del GPS), con el último cuarto menor: los caballos desaceleran al final.

# COMMAND ----------

display(gold.select("speed_q1", "speed_q2", "speed_q3", "speed_q4", "late_speed_ratio", "early_speed_ratio",
                    "time_to_top_speed_frac", "speed_cv", "n_decel_events", "max_decel_kmh",
                    "max_speed_kmh", "raw_max_speed_kmh", "n_gps_spikes", "n_frames", "race_duration_s").describe())

raros = gold.filter(
    (F.least("speed_q1", "speed_q2", "speed_q3", "speed_q4") < 30)
    | F.col("speed_q1").isNull() | F.col("speed_q4").isNull())
con_saltos = gold.filter(F.col("n_gps_spikes") > 0).count()
print(f"Caballo-carrera con al menos un salto de GPS excluido: {con_saltos:,} de {n_rows:,} "
      f"({100 * con_saltos / n_rows:.1f} %)")
print(f"Caballo-carrera con algún cuarto < 30 km/h o vacío (posible caballo detenido): {raros.count():,}")
display(raros.orderBy("speed_q4").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC **Comentario sobre valores raros** (ajústalo con lo que muestren las celdas anteriores):
# MAGIC
# MAGIC - **Saltos de GPS:** 1.588 frames (0,03 %) superan 90 km/h, con un máximo de 3.502 km/h. Ahora se excluyen y
# MAGIC   se cuentan en `n_gps_spikes`; la celda anterior muestra cuántos caballo-carrera tenían al menos uno.
# MAGIC - **Último cuarto muy lento o en 0:** caballos que **se detuvieron o fueron frenados** durante la carrera, o
# MAGIC   tracking que siguió registrando después de la llegada (el caballo trotando de vuelta).
# MAGIC - **Velocidades medias altas (~67–71 km/h en los cuartos centrales, mediana de frames 68,5):** el ruido normal del
# MAGIC   GPS hace que la trayectoria "zigzaguee" y que la distancia Haversine entre frames salga un poco mayor que la
# MAGIC   real, así que las velocidades absolutas están algo infladas. Las features de **forma** (ratios) son mucho
# MAGIC   menos sensibles a ese sesgo, porque se dividen por el promedio del propio caballo.
# MAGIC
# MAGIC Estos casos **no se borran** de Gold: los que siguen siendo raros son justo los que el modelo de la Parte 2
# MAGIC debería marcar como anómalos.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5 · Escritura como tabla administrada

# COMMAND ----------

(gold.write.format("delta")
     .mode("overwrite")
     .option("overwriteSchema", "true")
     .saveAsTable(T_GOLD_RACE_DYNAMICS))

spark.sql(f"COMMENT ON TABLE {T_GOLD_RACE_DYNAMICS} IS "
          "'Perfil de velocidad por caballo-carrera (grano: START_KEYS). Features de forma para clustering de "
          "estilos de carrera y detección de anomalías. Fuente: silver_derby_tracking.'")
print(f"✓ {T_GOLD_RACE_DYNAMICS}: {spark.table(T_GOLD_RACE_DYNAMICS).count():,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificación

# COMMAND ----------

display(spark.table(T_GOLD_RACE_DYNAMICS).orderBy(*START_KEYS).limit(20))