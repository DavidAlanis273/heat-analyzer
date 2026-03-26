# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 - Feature Engineering
# MAGIC Toma los datos limpios del notebook 01, calcula métricas de rendimiento
# MAGIC por termopar, y genera perfiles de cada calefactor.

# COMMAND ----------

import os
import sys
import pandas as pd
import numpy as np

repo_root = '/Workspace/Users/david.alanis@watlow.com/heat-analyzer'
sys.path.insert(0, repo_root)

from utils.features import add_rolling_features, compute_heater_profiles
from utils.reader import get_thermocouple_columns
from config.settings import CLEAN_DATA_FILE, FEATURES_FILE, PROFILES_FILE, OUTPUT_DIR, ROLLING_WINDOW

print(f"Repo root: {repo_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1: Cargar datos limpios

# COMMAND ----------

clean_path = os.path.join(repo_root, CLEAN_DATA_FILE)
df = pd.read_csv(clean_path)

print(f"Datos cargados: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Heaters: {df['heater_id'].unique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2: Agregar rolling features por heater y termopar

# COMMAND ----------

tc_columns = get_thermocouple_columns(df)
print(f"Thermocouples found: {len(tc_columns)}")
print(f"Rolling window: {ROLLING_WINDOW} readings ({ROLLING_WINDOW * 5} seconds)")

# COMMAND ----------

# Procesar cada heater por separado para que el rolling no cruce entre archivos
all_heaters = []

for heater_id in df['heater_id'].unique():
    print(f"\n  Processing: {heater_id}")
    hdf = df[df['heater_id'] == heater_id].copy()
    
    # Agregar rolling features a cada termopar activo
    active_tcs = [tc for tc in tc_columns if hdf[tc].notna().any()]
    for tc in active_tcs:
        hdf = add_rolling_features(hdf, tc, window=ROLLING_WINDOW)
    
    print(f"    Readings: {len(hdf)} | Active TCs: {len(active_tcs)} | New columns: {len(hdf.columns) - len(df.columns)}")
    all_heaters.append(hdf)

df_features = pd.concat(all_heaters, ignore_index=True)
print(f"\nDataFrame with features: {df_features.shape[0]} rows x {df_features.shape[1]} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3: Generar perfiles por termopar

# COMMAND ----------

all_profiles = []

for heater_id in df['heater_id'].unique():
    print(f"\n  Profiling: {heater_id}")
    hdf = df[df['heater_id'] == heater_id].copy()
    
    profiles = compute_heater_profiles(hdf, tc_columns, heater_id)
    all_profiles.append(profiles)
    
    print(f"    Thermocouples profiled: {len(profiles)}")

df_profiles = pd.concat(all_profiles, ignore_index=True)
print(f"\nTotal profiles generated: {len(df_profiles)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4: Explorar los perfiles

# COMMAND ----------

# Mostrar perfiles del primer heater como ejemplo
first_heater = df_profiles['heater_id'].unique()[0]
example = df_profiles[df_profiles['heater_id'] == first_heater]

print(f"Example profiles for: {first_heater}\n")
for _, row in example.iterrows():
    print(f"  {row['thermocouple']}:")
    print(f"    Temp range: {row['temp_min']}°C to {row['temp_max']}°C (avg: {row['temp_mean']}°C)")
    print(f"    Heating rate: {row['heating_rate_c_per_min']}°C/min")
    print(f"    Stability: {row['stability']} | Overshoot: {row['overshoot']}°C")
    print()

# COMMAND ----------

# Resumen comparativo entre heaters
print("Comparative summary across heaters:\n")
summary = df_profiles.groupby('heater_id').agg(
    num_tcs=('thermocouple', 'count'),
    avg_temp_max=('temp_max', 'mean'),
    avg_heating_rate=('heating_rate_c_per_min', 'mean'),
    avg_stability=('stability', 'mean'),
    max_overshoot=('overshoot', 'max')
).round(4)

for heater_id, row in summary.iterrows():
    print(f"  {heater_id}:")
    print(f"    TCs: {row['num_tcs']} | Avg max temp: {row['avg_temp_max']}°C")
    print(f"    Avg heating rate: {row['avg_heating_rate']}°C/min | Avg stability: {row['avg_stability']}")
    print(f"    Max overshoot: {row['max_overshoot']}°C")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5: Guardar resultados

# COMMAND ----------

output_path = os.path.join(repo_root, OUTPUT_DIR)
os.makedirs(output_path, exist_ok=True)

# Guardar datos con features
features_path = os.path.join(repo_root, FEATURES_FILE)
df_features.to_csv(features_path, index=False)
print(f"Features saved: {features_path}")
print(f"  Rows: {len(df_features)} | Columns: {len(df_features.columns)}")

# Guardar perfiles
profiles_path = os.path.join(repo_root, PROFILES_FILE)
df_profiles.to_csv(profiles_path, index=False)
print(f"\nProfiles saved: {profiles_path}")
print(f"  Rows: {len(df_profiles)} | Columns: {len(df_profiles.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificación final

# COMMAND ----------

df_verify = pd.read_csv(profiles_path)
print(f"Profiles verification: {len(df_verify)} thermocouple profiles across {df_verify['heater_id'].nunique()} heaters")
print(f"\nColumns: {list(df_verify.columns)}")
df_verify.head(10)