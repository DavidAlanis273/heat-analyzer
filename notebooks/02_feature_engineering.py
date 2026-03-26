# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 - Feature Engineering
# MAGIC Calcula métricas de rendimiento POR ARCHIVO y POR TERMOPAR.

# COMMAND ----------

import os
import sys
import importlib
import pandas as pd
import numpy as np

repo_root = '/Workspace/Users/david.alanis@watlow.com/heat-analyzer'
sys.path.insert(0, repo_root)

import utils.reader
importlib.reload(utils.reader)

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

tc_columns = get_thermocouple_columns(df)
print(f"Datos cargados: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Thermocouples: {len(tc_columns)} -> {tc_columns}")
print(f"Heaters: {list(df['heater_id'].unique())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2: Resumen individual por archivo
# MAGIC Cada archivo es una prueba independiente. Los resultados se muestran por separado.

# COMMAND ----------

for heater_id in df['heater_id'].unique():
    hdf = df[df['heater_id'] == heater_id]
    duration_sec = hdf['elapsed_seconds'].max()
    duration_min = duration_sec / 60
    duration_hrs = duration_min / 60
    
    active_tcs = [tc for tc in tc_columns if hdf[tc].notna().any()]
    missing_tcs = [tc for tc in tc_columns if hdf[tc].isna().all()]
    
    print(f"\n{'='*60}")
    print(f"  TEST: {heater_id}")
    print(f"{'='*60}")
    print(f"  Duration:     {duration_min:.1f} minutes ({duration_hrs:.1f} hours)")
    print(f"  Readings:     {len(hdf)}")
    print(f"  Interval:     ~{duration_sec / max(len(hdf)-1, 1):.0f} seconds between readings")
    print(f"  Active TCs:   {len(active_tcs)} of {len(tc_columns)}")
    
    if missing_tcs:
        print(f"  Missing TCs:  {missing_tcs} (no data)")
    
    # Rango de temperatura por TC
    print(f"\n  Temperature summary per thermocouple:")
    print(f"  {'TC':<6} {'Start':>8} {'End':>8} {'Min':>8} {'Max':>8} {'Avg':>8} {'Range':>8}")
    print(f"  {'─'*54}")
    
    for tc in active_tcs:
        series = hdf[tc].dropna()
        print(f"  {tc:<6} {series.iloc[0]:>7.1f}° {series.iloc[-1]:>7.1f}° {series.min():>7.1f}° {series.max():>7.1f}° {series.mean():>7.1f}° {series.max()-series.min():>7.1f}°")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3: Agregar rolling features

# COMMAND ----------

all_heaters = []

for heater_id in df['heater_id'].unique():
    print(f"\n  Processing: {heater_id}")
    hdf = df[df['heater_id'] == heater_id].copy()
    
    active_tcs = [tc for tc in tc_columns if hdf[tc].notna().any()]
    for tc in active_tcs:
        hdf = add_rolling_features(hdf, tc, window=ROLLING_WINDOW)
    
    print(f"    Active TCs: {len(active_tcs)} | New columns added: {len(hdf.columns) - len(df.columns)}")
    all_heaters.append(hdf)

df_features = pd.concat(all_heaters, ignore_index=True)
print(f"\nDataFrame with features: {df_features.shape[0]} rows x {df_features.shape[1]} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4: Perfiles detallados por termopar, por archivo

# COMMAND ----------

all_profiles = []

for heater_id in df['heater_id'].unique():
    hdf = df[df['heater_id'] == heater_id].copy()
    profiles = compute_heater_profiles(hdf, tc_columns, heater_id)
    all_profiles.append(profiles)

df_profiles = pd.concat(all_profiles, ignore_index=True)
print(f"Total profiles: {len(df_profiles)}")

# COMMAND ----------

# Mostrar perfiles organizados por archivo
for heater_id in df_profiles['heater_id'].unique():
    hp = df_profiles[df_profiles['heater_id'] == heater_id].copy()
    hp = hp.sort_values('temp_max', ascending=False)
    
    print(f"\n{'='*70}")
    print(f"  PERFORMANCE PROFILE: {heater_id}")
    print(f"{'='*70}")
    print(f"\n  {'TC':<6} {'Max°C':>7} {'Avg°C':>7} {'Rate':>10} {'Stability':>10} {'Overshoot':>10}")
    print(f"  {'─'*52}")
    
    for _, row in hp.iterrows():
        rate_str = f"{row['heating_rate_c_per_min']:.3f}°/min"
        stab_str = f"{row['stability']:.3f}"
        over_str = f"{row['overshoot']:.1f}°C"
        print(f"  {row['thermocouple']:<6} {row['temp_max']:>6.1f} {row['temp_mean']:>7.1f} {rate_str:>10} {stab_str:>10} {over_str:>10}")
    
    # Identificar termopares con comportamiento especial
    avg_max = hp['temp_max'].mean()
    hot_tcs = hp[hp['temp_max'] > avg_max * 1.3]['thermocouple'].tolist()
    cold_tcs = hp[hp['temp_max'] < avg_max * 0.5]['thermocouple'].tolist()
    stable_tcs = hp.nsmallest(3, 'stability')['thermocouple'].tolist()
    unstable_tcs = hp.nlargest(3, 'stability')['thermocouple'].tolist()
    
    print(f"\n  Observations:")
    if hot_tcs:
        print(f"    Hottest TCs (>30% above avg): {', '.join(hot_tcs)}")
    if cold_tcs:
        print(f"    Coldest TCs (<50% of avg):    {', '.join(cold_tcs)}")
    print(f"    Most stable:  {', '.join(stable_tcs)}")
    print(f"    Least stable: {', '.join(unstable_tcs)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5: Guardar resultados

# COMMAND ----------

output_path = os.path.join(repo_root, OUTPUT_DIR)
os.makedirs(output_path, exist_ok=True)

features_path = os.path.join(repo_root, FEATURES_FILE)
df_features.to_csv(features_path, index=False)
print(f"Features saved: {features_path} ({len(df_features)} rows x {len(df_features.columns)} cols)")

profiles_path = os.path.join(repo_root, PROFILES_FILE)
df_profiles.to_csv(profiles_path, index=False)
print(f"Profiles saved: {profiles_path} ({len(df_profiles)} profiles)")