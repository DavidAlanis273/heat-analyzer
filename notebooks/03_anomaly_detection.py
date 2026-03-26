# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install scikit-learn --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 - Detección de Anomalías
# MAGIC Aplica Z-Score, IQR, Isolation Forest y detección de lecturas congeladas
# MAGIC a cada termopar de cada calefactor.


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

from utils.detection import run_all_detection
from utils.reader import get_thermocouple_columns
from config.settings import (
    FEATURES_FILE, ANOMALIES_FILE, OUTPUT_DIR,
    ZSCORE_THRESHOLD, IQR_MULTIPLIER, 
    ISOLATION_FOREST_CONTAMINATION, FROZEN_WINDOW
)

# Parámetros de detección
detection_settings = {
    'zscore_threshold': ZSCORE_THRESHOLD,
    'iqr_multiplier': IQR_MULTIPLIER,
    'iforest_contamination': ISOLATION_FOREST_CONTAMINATION,
    'frozen_window': FROZEN_WINDOW
}

print(f"Detection settings: {detection_settings}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1: Cargar datos con features

# COMMAND ----------

features_path = os.path.join(repo_root, FEATURES_FILE)
df = pd.read_csv(features_path)

print(f"Datos cargados: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Heaters: {df['heater_id'].unique()}")

tc_columns = get_thermocouple_columns(df)
print(f"Thermocouples: {len(tc_columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2: Correr detección de anomalías por heater y termopar

# COMMAND ----------

all_anomalies = []

for heater_id in df['heater_id'].unique():
    print(f"\n  Analyzing: {heater_id}")
    hdf = df[df['heater_id'] == heater_id].copy()
    
    active_tcs = [tc for tc in tc_columns if hdf[tc].notna().any()]
    heater_anomaly_count = 0
    
    for tc in active_tcs:
        results = run_all_detection(hdf, tc, detection_settings)
        results['heater_id'] = heater_id
        
        tc_anomalies = results['anomaly_consensus'].sum()
        heater_anomaly_count += tc_anomalies
        
        # Solo guardar filas donde al menos un método detectó algo
        anomaly_rows = results[results['anomaly_count'] > 0]
        if len(anomaly_rows) > 0:
            all_anomalies.append(anomaly_rows)
    
    print(f"    Active TCs: {len(active_tcs)} | Consensus anomalies: {heater_anomaly_count}")

df_anomalies = pd.concat(all_anomalies, ignore_index=True)
print(f"\nTotal anomaly records: {len(df_anomalies)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3: Resumen de anomalías por método

# COMMAND ----------

print("Anomalies detected by method:\n")

methods = {
    'Z-Score': 'anomaly_zscore',
    'IQR': 'anomaly_iqr',
    'Isolation Forest': 'anomaly_iforest',
    'Frozen Readings': 'anomaly_frozen',
    'Consensus (2+ methods)': 'anomaly_consensus'
}

for name, col in methods.items():
    count = df_anomalies[col].sum()
    print(f"  {name}: {int(count)} anomalies")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4: Anomalías por heater

# COMMAND ----------

print("Anomalies by heater:\n")

for heater_id in df_anomalies['heater_id'].unique():
    hdf = df_anomalies[df_anomalies['heater_id'] == heater_id]
    consensus = hdf['anomaly_consensus'].sum()
    zscore = hdf['anomaly_zscore'].sum()
    iqr = hdf['anomaly_iqr'].sum()
    iforest = hdf['anomaly_iforest'].sum()
    frozen = hdf['anomaly_frozen'].sum()
    tcs_affected = hdf[hdf['anomaly_consensus']]['thermocouple'].nunique()
    
    print(f"  {heater_id}:")
    print(f"    Consensus: {int(consensus)} | Z-Score: {int(zscore)} | IQR: {int(iqr)} | iForest: {int(iforest)} | Frozen: {int(frozen)}")
    print(f"    Thermocouples affected: {tcs_affected}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5: Top anomalías más severas

# COMMAND ----------

# Las anomalías detectadas por más métodos son las más confiables
severe = df_anomalies[df_anomalies['anomaly_count'] >= 3].copy()
severe = severe.sort_values('anomaly_count', ascending=False)

print(f"Severe anomalies (detected by 3+ methods): {len(severe)}\n")

if len(severe) > 0:
    for heater_id in severe['heater_id'].unique():
        hsev = severe[severe['heater_id'] == heater_id]
        print(f"  {heater_id}: {len(hsev)} severe anomalies")
        
        # Mostrar top 5 por heater
        for _, row in hsev.head(5).iterrows():
            print(f"    {row['thermocouple']} at {row['elapsed_seconds']}s: {row['temperature']:.2f}°C (detected by {int(row['anomaly_count'])} methods)")
        
        if len(hsev) > 5:
            print(f"    ... and {len(hsev) - 5} more")
        print()
else:
    print("  No severe anomalies found — all detections are from 1-2 methods only.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6: Anomalías por termopar (¿cuáles sensores tienen más problemas?)

# COMMAND ----------

tc_anomaly_summary = df_anomalies[df_anomalies['anomaly_consensus']].groupby('thermocouple').agg(
    total_anomalies=('anomaly_consensus', 'sum'),
    heaters_affected=('heater_id', 'nunique'),
    avg_temp=('temperature', 'mean')
).sort_values('total_anomalies', ascending=False)

print("Thermocouples with most consensus anomalies:\n")
for tc, row in tc_anomaly_summary.head(10).iterrows():
    print(f"  {tc}: {int(row['total_anomalies'])} anomalies across {int(row['heaters_affected'])} heater(s) | Avg temp: {row['avg_temp']:.1f}°C")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 7: Guardar resultados

# COMMAND ----------

output_path = os.path.join(repo_root, OUTPUT_DIR)
os.makedirs(output_path, exist_ok=True)

anomalies_path = os.path.join(repo_root, ANOMALIES_FILE)
df_anomalies.to_csv(anomalies_path, index=False)

print(f"Anomalies saved: {anomalies_path}")
print(f"  Total records: {len(df_anomalies)}")
print(f"  Consensus anomalies: {int(df_anomalies['anomaly_consensus'].sum())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen ejecutivo

# COMMAND ----------

total_readings = df.shape[0] * len(tc_columns)
total_consensus = int(df_anomalies['anomaly_consensus'].sum())
pct = (total_consensus / total_readings) * 100 if total_readings > 0 else 0

print("=" * 60)
print("   ANOMALY DETECTION - EXECUTIVE SUMMARY")
print("=" * 60)
print(f"   Total readings analyzed:    {total_readings:,}")
print(f"   Consensus anomalies found:  {total_consensus:,} ({pct:.2f}%)")
print(f"   Heaters analyzed:           {df['heater_id'].nunique()}")
print(f"   Thermocouples per heater:   {len(tc_columns)}")
print()
print("   By method:")
for name, col in methods.items():
    count = int(df_anomalies[col].sum())
    print(f"     {name}: {count:,}")
print("=" * 60)