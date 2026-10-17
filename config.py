# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Configuración centralizada del pipeline Big Data Derby 2022
# MAGIC
# MAGIC Este notebook se importa con `%run ../00-setup/config`.
# MAGIC Centraliza paths, schemas, nombres de tabla y parámetros del pipeline.
# MAGIC
# MAGIC ## Arquitectura real del curso
# MAGIC
# MAGIC - **Catálogo compartido**: `20080_databricks`, ya creado por el profesor.
# MAGIC   El estudiante NO crea catálogos ni schemas: solo necesita `USE CATALOG`
# MAGIC   sobre el catálogo y ser dueño de su schema, que también crea el profesor.
# MAGIC   El nombre del catálogo cambia entre iteraciones del curso: confírmalo con
# MAGIC   tu docente antes de correr nada. Como empieza por dígito, en SQL siempre
# MAGIC   va entre backticks.
# MAGIC - **Un schema por estudiante**: `<CATALOG>.<SCHEMA>`, donde `SCHEMA` es tu
# MAGIC   `nombre_apellido` en snake_case (por ejemplo `camilo_soto`). Es lo ÚNICO
# MAGIC   que debes editar en este notebook.
# MAGIC - **Landing EN TU PROPIO Volume**: a diferencia del módulo 06 (que lee de un
# MAGIC   landing compartido de solo lectura), aquí **cada estudiante descarga el
# MAGIC   dataset de Kaggle a su propio Volume** `datos`, bajo `landing/derby`. Lo
# MAGIC   hace el notebook `00-setup/02_download_dataset.py`.
# MAGIC - **Capas medallion como tablas administradas de Unity Catalog**: bronze,
# MAGIC   silver y gold se escriben con `saveAsTable` en `<CATALOG>.<SCHEMA>`. UC
# MAGIC   gestiona el almacenamiento, así que las tablas se consultan por nombre
# MAGIC   desde SQL, aparecen en Catalog Explorer con linaje y admiten `GRANT` por
# MAGIC   tabla. Las capas se distinguen por prefijo en el nombre (`bronze_derby_`,
# MAGIC   `silver_derby_`, `gold_derby_`).

# COMMAND ----------

# =============================================================================
# Unity Catalog — Catálogo, Schema y Volume
# =============================================================================
# ── Bloque editable por el estudiante ───────────────────────────────────────
# SOLO debes cambiar SCHEMA por tu nombre_apellido en snake_case.
# CATALOG lo crea el profesor: no lo inventes, confirma con tu docente el del
# semestre. VOLUME no se cambia: `datos` es el Volume que ya existe en el
# schema de cada estudiante.
CATALOG = "51810_databricks" # lo crea el profesor; confírmalo con tu docente
SCHEMA =  "sebastianc_munoz"     # ← CAMBIAR: tu nombre_apellido en snake_case
VOLUME =  "landing"    

# =============================================================================
# Volume del estudiante: landing (descarga de Kaggle) y nada más
# =============================================================================
# En Unity Catalog la ruta de un Volume es /Volumes/<catalogo>/<schema>/<volume>/...
# Las capas medallion NO viven aquí: son tablas administradas (ver más abajo).
DATALAKE_BASE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# Landing EN TU Volume: aquí baja el dataset cada estudiante (no es compartido).
LANDING_BASE = f"{DATALAKE_BASE}/landing/derby"

# =============================================================================
# Dataset de Kaggle
# =============================================================================
# Slug de la competencia en Kaggle. La descarga la hace 02_download_dataset.py
# con la librería kagglehub (kagglehub.competition_download(KAGGLE_COMPETITION)).
KAGGLE_COMPETITION = "big-data-derby-2022"

# Nombres de los CSV tal como vienen en el ZIP de Kaggle.
FILE_RACE = "nyra_race_table.csv"
FILE_START = "nyra_start_table.csv"
FILE_TRACKING = "nyra_tracking_table.csv"
FILE_COMPLETE = "nyra_2019_complete.csv"  # versión desnormalizada; no se ingesta

# Rutas completas de cada CSV dentro de tu landing
PATH_RACE_CSV = f"{LANDING_BASE}/{FILE_RACE}"
PATH_START_CSV = f"{LANDING_BASE}/{FILE_START}"
PATH_TRACKING_CSV = f"{LANDING_BASE}/{FILE_TRACKING}"

# =============================================================================
# Tablas administradas de Unity Catalog (catálogo.schema.tabla)
# =============================================================================
# Todas las capas viven en tu schema. El catálogo empieza por dígito, así que
# va entre backticks; estos nombres sirven tal cual en spark.table(),
# saveAsTable(), DeltaTable.forName() y dentro de spark.sql().
#
# Otros módulos escriben en este mismo schema (bronze_yellow_trips de 06,
# bronze_travel_times de 04, ...), por eso los nombres de este módulo llevan
# `derby_` para no chocar.
TABLE_PREFIX = f"`{CATALOG}`.`{SCHEMA}`"

# ── Bronze — datos crudos tal cual llegan ──────────────────────────────────
T_BRONZE_RACE = f"{TABLE_PREFIX}.bronze_derby_race"
T_BRONZE_START = f"{TABLE_PREFIX}.bronze_derby_start"
T_BRONZE_TRACKING = f"{TABLE_PREFIX}.bronze_derby_tracking"

# ── Silver — datos limpios, unidos y enriquecidos ──────────────────────────
T_SILVER_RACES = f"{TABLE_PREFIX}.silver_derby_races"            # carrera + nivel carrera
T_SILVER_STARTS = f"{TABLE_PREFIX}.silver_derby_starts"          # caballo-carrera (start+race)
T_SILVER_STARTS_REJECTED = f"{TABLE_PREFIX}.silver_derby_starts_rejected"
T_SILVER_TRACKING = f"{TABLE_PREFIX}.silver_derby_tracking"      # tracking con features de frame

# ── Gold — KPIs de negocio y features de ML ────────────────────────────────
T_GOLD_RACE_SUMMARY = f"{TABLE_PREFIX}.gold_derby_race_summary"
T_GOLD_TRACK_METRICS = f"{TABLE_PREFIX}.gold_derby_track_metrics"
T_GOLD_JOCKEY_PERFORMANCE = f"{TABLE_PREFIX}.gold_derby_jockey_performance"
T_GOLD_ML_FEATURES = f"{TABLE_PREFIX}.gold_derby_ml_features"
T_GOLD_RACE_DYNAMICS = f"{TABLE_PREFIX}.gold_derby_race_dynamics"
T_GOLD_RACE_PROFILES = f"{TABLE_PREFIX}.gold_derby_race_profiles"

# ── ML — predicciones batch ────────────────────────────────────────────────
T_ML_PREDICTIONS = f"{TABLE_PREFIX}.gold_derby_ml_predictions"

# Iniciales del estudiante (derivadas del schema)
USER_INITIALS = SCHEMA.upper() if SCHEMA else "STU"

# =============================================================================
# Modelo registrado en Unity Catalog
# =============================================================================
# Va en tu mismo schema: eres su dueño, así que tienes CREATE MODEL sin grants
# extra (igual que en el módulo 06).
MODEL_NAME = f"{CATALOG}.{SCHEMA}.derby_top3_gbt"

# Directorio temporal de MLflow para guardar/cargar modelos Spark MLlib. En
# compute serverless o de acceso estándar, mlflow.spark exige un Volume de
# Unity Catalog para esto (ver 00-setup/config.py del módulo 06).
MLFLOW_DFS_TMPDIR = f"{DATALAKE_BASE}/_mlflow_tmp/derby"

# COMMAND ----------

# =============================================================================
# Esquemas de los CSV del landing — BRONZE = TODO STRING
# =============================================================================
# Convención medallion de este módulo: en Bronze todo entra como texto (string),
# tal cual viene del CSV. El casteo a tipos reales (date, int, double) se hace en
# Silver, que es donde viven las reglas de calidad. Así Bronze es fiel al origen
# y ninguna fila se pierde ni rompe la ingesta por un valor mal formado (p.ej.
# post_time = "01220", que no es una hora parseable).
#
# Spark mapea por POSICIÓN cuando se pasa un schema a spark.read.csv, así que el
# orden de los campos debe coincidir con el del archivo. Diccionario oficial de la
# competencia Big Data Derby 2022 (Kaggle).
from pyspark.sql.types import StructType, StructField, StringType


def _all_string_schema(col_names):
    """Construye un StructType con todas las columnas como StringType (Bronze)."""
    return StructType([StructField(name, StringType()) for name in col_names])


# ── Columnas de cada CSV, EN ORDEN (así vienen en el archivo) ───────────────
RACE_COLUMNS = [
    "track_id",        # AQU / BEL / SAR
    "race_date",       # fecha (texto YYYY-MM-DD en el CSV)
    "race_number",
    "distance_id",     # distancia en furlongs ×100 (600 = 6 furlongs)
    "course_type",     # 1 char: M/D/O/I/T
    "track_condition", # 3 chars: YL/FM/SY/GD/FT/MY/SF
    "run_up_distance", # pies
    "race_type",       # STK / ALW / CLM ...
    "purse",           # dólares
    "post_time",       # código de hora (texto, no parseable como reloj)
]

START_COLUMNS = [
    "track_id",
    "race_date",
    "race_number",
    "program_number",     # alfanumérico (p.ej. "1", "1A"); trae espacios
    "weight_carried",     # libras
    "jockey",
    "odds",               # cuota escalada ×100 (p.ej. 250 = 2.50)
    "position_at_finish",
]

TRACKING_COLUMNS = [
    "track_id",
    "race_date",
    "race_number",
    "program_number",
    "trakus_index",       # orden del frame en la carrera
    "latitude",
    "longitude",
]

# Esquemas Bronze: todo string
RACE_SCHEMA = _all_string_schema(RACE_COLUMNS)
START_SCHEMA = _all_string_schema(START_COLUMNS)
TRACKING_SCHEMA = _all_string_schema(TRACKING_COLUMNS)

# ── Tipos objetivo para el casteo en SILVER ─────────────────────────────────
# Silver castea con try_cast (null-safe): un valor que no castee queda null y lo
# atrapan las reglas de calidad, en vez de romper la ingesta.
RACE_CASTS = {
    "race_date":       "date",
    "race_number":     "int",
    "distance_id":     "int",
    "run_up_distance": "int",
    "purse":           "double",
}
START_CASTS = {
    "race_number":        "int",
    "weight_carried":     "int",
    "odds":               "double",
    "position_at_finish": "int",
}
TRACKING_CASTS = {
    "race_number":  "int",
    "trakus_index": "int",
    "latitude":     "double",
    "longitude":    "double",
}

# Claves de negocio
RACE_KEYS = ["track_id", "race_date", "race_number"]          # identifica una carrera
START_KEYS = RACE_KEYS + ["program_number"]                   # identifica un caballo-carrera
TRACKING_KEYS = START_KEYS + ["trakus_index"]                 # identifica un frame

# =============================================================================
# Diccionarios de códigos (del data dictionary oficial de Kaggle)
# =============================================================================
# Silver mapea estos códigos a etiquetas legibles en columnas *_desc, dejando
# también el código original.

# Hipódromo (3 chars)
TRACK_NAMES = {
    "AQU": "Aqueduct",
    "BEL": "Belmont",
    "SAR": "Saratoga",
}
VALID_TRACK_IDS = list(TRACK_NAMES.keys())

# course_type: 1 carácter → tipo de pista
COURSE_TYPE_NAMES = {
    "M": "Hurdle",
    "D": "Dirt",
    "O": "Outer turf",
    "I": "Inner turf",
    "T": "Turf",
}

# track_condition: 3 chars (vienen con espacios de relleno) → condición de pista
TRACK_CONDITION_NAMES = {
    "YL": "Yielding",
    "FM": "Firm",
    "SY": "Sloppy",
    "GD": "Good",
    "FT": "Fast",
    "MY": "Muddy",
    "SF": "Soft",
}

# race_type: 5 chars → clasificación de la carrera
RACE_TYPE_NAMES = {
    "STK": "Stakes",
    "WCL": "Waiver Claiming",
    "WMC": "Waiver Maiden Claiming",
    "SST": "Starter Stakes",
    "SHP": "Starter Handicap",
    "CLM": "Claiming",
    "STR": "Starter Allowance",
    "AOC": "Allowance Optional Claimer",
    "SOC": "Starter Optional Claimer",
    "MCL": "Maiden Claiming",
    "ALW": "Allowance",
    "MSW": "Maiden Special Weight",
}

# Frecuencia de muestreo del tracking (Hz). ~4 Hz según la documentación de la
# competencia: cada trakus_index son ~0.25 segundos.
TRACKING_HZ = 4.0

# COMMAND ----------

def read_csv_with_schema(spark, path, schema):
    """Lee un CSV del landing con esquema explícito (Bronze = todo string).

    Agrega la columna técnica `_source_file` con la ruta del archivo, igual que
    en el módulo 06, para trazabilidad en bronze. No castea tipos: eso es tarea
    de Silver.
    """
    from pyspark.sql import functions as F
    return (
        spark.read
        .option("header", True)
        .schema(schema)
        .csv(path)
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )


def apply_casts(df, casts):
    """Castea columnas string → su tipo objetivo de forma null-safe (para Silver).

    Usa la función SQL `try_cast`, que devuelve NULL en vez de fallar cuando un
    valor no se puede convertir. Así un valor mal formado en Bronze no rompe
    Silver: queda null y lo atrapan las reglas de calidad.

    `casts` es un dict {columna: tipo_sql}, p.ej. {"race_number": "int"}.
    """
    from pyspark.sql import functions as F
    for col_name, target_type in casts.items():
        df = df.withColumn(col_name, F.expr(f"try_cast(`{col_name}` AS {target_type})"))
    return df


def map_code_column(source_col, mapping):
    """Devuelve una Column que traduce un código a su etiqueta legible.

    `mapping` es un dict {codigo: etiqueta}. Un código no listado queda con su
    valor original (así no se pierde información si aparece un código nuevo).
    Pensado para course_type, track_condition, race_type, track_id.
    """
    from pyspark.sql import functions as F
    col = F.col(source_col) if isinstance(source_col, str) else source_col
    result = col  # default: el valor original
    for code, label in mapping.items():
        result = F.when(col == code, F.lit(label)).otherwise(result)
    return result


def post_time_to_hhmm(post_time_col):
    """Convierte post_time (texto, p.ej. '01220') a hora legible 'HH:mm'.

    Según el data dictionary, post_time es la hora de salida en 5 caracteres con
    un cero de relleno al inicio; los ÚLTIMOS 4 dígitos son HHMM (reloj de 12 h,
    sin AM/PM: una carrera a las '0130' es la 1:30 de la tarde). Ejemplos reales:
      '01220' → '1220' → 12:20
      '00130' → '0130' → 01:30
      '00445' → '0445' → 04:45
    Devuelve una Column string 'HH:mm'. Si el valor no tiene 4 dígitos usables o
    la hora/minuto quedan fuera de rango, queda null.
    """
    from pyspark.sql import functions as F
    pt = post_time_col if not isinstance(post_time_col, str) else F.col(post_time_col)
    # Solo dígitos; tomamos los ÚLTIMOS 4 (HHMM). NO usar lpad(.,4): en Spark lpand
    # trunca por la izquierda si el string es más largo, lo que corrompe el valor.
    digits = F.regexp_replace(pt, r"\D", "")
    last4 = F.substring(digits, -4, 4)   # últimos 4 chars = HHMM
    hh = F.substring(last4, 1, 2)
    mm = F.substring(last4, 3, 2)
    hhmm = F.concat_ws(":", hh, mm)
    return F.when(
        (F.length(digits) >= 4)
        & (hh.cast("int").between(0, 23))
        & (mm.cast("int").between(0, 59)),
        hhmm,
    )

# COMMAND ----------

print(f"Catalog:            {CATALOG}")
print(f"Schema:             {SCHEMA}")
print(f"Volume:             {VOLUME}")
print(f"Landing (tuyo):     {LANDING_BASE}")
print(f"Kaggle competition: {KAGGLE_COMPETITION}")
print("")
print(f"Bronze race:        {T_BRONZE_RACE}")
print(f"Bronze start:       {T_BRONZE_START}")
print(f"Bronze tracking:    {T_BRONZE_TRACKING}")
print(f"Silver races:       {T_SILVER_RACES}")
print(f"Silver starts:      {T_SILVER_STARTS}")
print(f"Silver tracking:    {T_SILVER_TRACKING}")
print(f"Gold race_summary:  {T_GOLD_RACE_SUMMARY}")
print(f"Gold track_metrics: {T_GOLD_TRACK_METRICS}")
print(f"Gold jockey_perf:   {T_GOLD_JOCKEY_PERFORMANCE}")
print(f"Gold ml_features:   {T_GOLD_ML_FEATURES}")
print(f"ML predicciones:    {T_ML_PREDICTIONS}")
print(f"Modelo UC:          {MODEL_NAME}")