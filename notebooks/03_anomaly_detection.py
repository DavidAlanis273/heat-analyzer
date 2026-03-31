# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install scikit-learn --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 - Anomalie detection
# MAGIC Detecta y contextualiza anomalías POR ARCHIVO y POR TERMOPAR.
# MAGIC Clasifica si ocurrieron durante rampa de calentamiento o estado estable.

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
S
from utils.detection import run_all_detection
from utils.reader import get_thermocouple_columns
from config.settings import (
    FEATURES_FILE, ANOMALIES_FILE, OUTPUT_DIR,
    ZSCORE_THRESHOLD, IQR_MULTIPLIER,
    ISOLATION_FOREST_CONTAMINATION, FROZEN_WINDOW
)

detection_settings = {
    'zscore_threshold': ZSCORE_THRESHOLD,
    'iqr_multiplier': IQR_MULTIPLIER,
    'iforest_contamination': ISOLATION_FOREST_CONTAMINATION,
    'frozen_window': FROZEN_WINDOW
}

print(f"Detection settings: {detection_settings}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load data with features

# COMMAND ----------

features_path = os.path.join(repo_root, FEATURES_FILE)
df = pd.read_csv(features_path)

tc_columns = get_thermocouple_columns(df)
print(f"Datos cargados: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Thermocouples: {len(tc_columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Detect anomalies by file and thermocouple

# COMMAND ----------

all_anomalies = []

for heater_id in df['heater_id'].unique():
    print(f"\n  Analyzing: {heater_id}")
    hdf = df[df['heater_id'] == heater_id].copy()
    active_tcs = [tc for tc in tc_columns if hdf[tc].notna().any()]
    
    for tc in active_tcs:
        results = run_all_detection(hdf, tc, detection_settings)
        results['heater_id'] = heater_id
        
        # Clasificar fase: rampa vs estado estable
        # Si rate_of_change promedio en una ventana es alto -> rampa
        roc_col = f"{tc}_rate_of_change"
        if roc_col in hdf.columns:
            roc_rolling = hdf[roc_col].abs().rolling(window=30, min_periods=1).mean()
            roc_threshold = roc_rolling.quantile(0.75)
            results['phase'] = 'ramping'
            stable_mask = hdf.loc[results.index, roc_col].abs().rolling(window=30, min_periods=1).mean() < roc_threshold
            results.loc[stable_mask[stable_mask].index, 'phase'] = 'steady_state'
        else:
            results['phase'] = 'unknown'
        
        anomaly_rows = results[results['anomaly_count'] > 0]
        if len(anomaly_rows) > 0:
            all_anomalies.append(anomaly_rows)
    
    heater_consensus = sum(r['anomaly_consensus'].sum() for r in all_anomalies if r['heater_id'].iloc[0] == heater_id) if all_anomalies else 0
    print(f"    Active TCs: {len(active_tcs)}")

df_anomalies = pd.concat(all_anomalies, ignore_index=True)
print(f"\nTotal anomaly records: {len(df_anomalies)}")
print(f"Consensus anomalies: {int(df_anomalies['anomaly_consensus'].sum())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Anomalie summary by file

# COMMAND ----------

for heater_id in df['heater_id'].unique():
    hdf_anom = df_anomalies[df_anomalies['heater_id'] == heater_id]
    consensus = hdf_anom[hdf_anom['anomaly_consensus'] == True]
    
    # Contar por fase
    ramping_count = len(consensus[consensus['phase'] == 'ramping']) if 'phase' in consensus.columns else 0
    steady_count = len(consensus[consensus['phase'] == 'steady_state']) if 'phase' in consensus.columns else 0
    
    print(f"\n{'='*65}")
    print(f"  ANOMALY REPORT: {heater_id}")
    print(f"{'='*65}")
    print(f"  Total consensus anomalies: {len(consensus)}")
    print(f"    During ramping phase:      {ramping_count} (expected — temperature changing rapidly)")
    print(f"    During steady state:       {steady_count} (these need attention)")
    print(f"  Thermocouples affected: {consensus['thermocouple'].nunique()}")
    
    print(f"\n  By method:")
    print(f"    Z-Score:          {int(hdf_anom['anomaly_zscore'].sum())}")
    print(f"    IQR:              {int(hdf_anom['anomaly_iqr'].sum())}")
    print(f"    Isolation Forest: {int(hdf_anom['anomaly_iforest'].sum())}")
    print(f"    Frozen Readings:  {int(hdf_anom['anomaly_frozen'].sum())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Thermocouple detail - what happened and why 

# COMMAND ----------

for heater_id in df['heater_id'].unique():
    hdf = df[df['heater_id'] == heater_id]
    hdf_anom = df_anomalies[df_anomalies['heater_id'] == heater_id]
    consensus = hdf_anom[hdf_anom['anomaly_consensus'] == True]
    
    if len(consensus) == 0:
        continue
    
    print(f"\n{'='*70}")
    print(f"  THERMOCOUPLE DETAIL: {heater_id}")
    print(f"{'='*70}")
    
    # Agrupar por TC
    tc_summary = consensus.groupby('thermocouple').agg(
        total_anomalies=('anomaly_consensus', 'sum'),
        avg_temp=('temperature', 'mean'),
        min_temp=('temperature', 'min'),
        max_temp=('temperature', 'max')
    ).sort_values('total_anomalies', ascending=False)
    
    for tc, row in tc_summary.iterrows():
        tc_consensus = consensus[consensus['thermocouple'] == tc]
        ramping = len(tc_consensus[tc_consensus['phase'] == 'ramping'])
        steady = len(tc_consensus[tc_consensus['phase'] == 'steady_state'])
        
        # Obtener datos completos del TC para contexto
        tc_full = hdf[tc].dropna()
        tc_avg_full = tc_full.mean()
        tc_max_full = tc_full.max()
        
        print(f"\n  {tc}:")
        print(f"    Full test avg: {tc_avg_full:.1f}°C | Max: {tc_max_full:.1f}°C")
        print(f"    Anomalies: {int(row['total_anomalies'])} total ({ramping} during ramping, {steady} during steady state)")
        print(f"    Anomaly temp range: {row['min_temp']:.1f}°C to {row['max_temp']:.1f}°C")
        
        # Diagnóstico automático
        if steady > ramping:
            print(f"    ⚠ Most anomalies in steady state — possible sensor issue or process instability")
        elif ramping > 0 and steady == 0:
            print(f"    ✓ All anomalies during ramping — normal behavior during temperature changes")
        
        # Detectar si es un TC "frío" comparado con los demás
        all_tc_maxes = [hdf[t].max() for t in tc_columns if t in hdf.columns and hdf[t].notna().any()]
        avg_all_maxes = np.mean(all_tc_maxes)
        if tc_max_full < avg_all_maxes * 0.5:
            print(f"    ℹ This TC reached only {tc_max_full:.0f}°C while average max was {avg_all_maxes:.0f}°C — likely positioned far from heat source")
        
        # Detectar si casi no se movió
        if tc_full.max() - tc_full.min() < 10:
            print(f"    ℹ Temperature range was only {tc_full.max() - tc_full.min():.1f}°C — sensor in isolated/ambient zone")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Severe anomalies (3+ methods agree)

# COMMAND ----------

severe = df_anomalies[df_anomalies['anomaly_count'] >= 3].copy()

for heater_id in df['heater_id'].unique():
    hsev = severe[severe['heater_id'] == heater_id]
    
    if len(hsev) == 0:
        print(f"\n  {heater_id}: No severe anomalies (good)")
        continue
    
    steady_severe = hsev[hsev['phase'] == 'steady_state']
    
    print(f"\n{'='*65}")
    print(f"  SEVERE ANOMALIES: {heater_id}")
    print(f"{'='*65}")
    print(f"  Total severe (3+ methods): {len(hsev)}")
    print(f"  During steady state: {len(steady_severe)} ← these matter most")
    
    if len(steady_severe) > 0:
        print(f"\n  Steady state severe anomalies:")
        for tc in steady_severe['thermocouple'].unique():
            tc_sev = steady_severe[steady_severe['thermocouple'] == tc]
            print(f"    {tc}: {len(tc_sev)} readings | Temp range: {tc_sev['temperature'].min():.1f}°C to {tc_sev['temperature'].max():.1f}°C")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Save results

# COMMAND ----------

output_path = os.path.join(repo_root, OUTPUT_DIR)
os.makedirs(output_path, exist_ok=True)

anomalies_path = os.path.join(repo_root, ANOMALIES_FILE)
df_anomalies.to_csv(anomalies_path, index=False)

total_readings = sum(len(df[df['heater_id'] == h]) * len([tc for tc in tc_columns if df[df['heater_id'] == h][tc].notna().any()]) for h in df['heater_id'].unique())
total_consensus = int(df_anomalies['anomaly_consensus'].sum())

print(f"Anomalies saved: {anomalies_path}")
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Total readings analyzed: {total_readings:,}")
print(f"  Consensus anomalies:     {total_consensus:,} ({total_consensus/max(total_readings,1)*100:.2f}%)")
print(f"  Heaters analyzed:        {df['heater_id'].nunique()}")
print(f"  Thermocouples:           {len(tc_columns)} per test")
print(f"{'='*60}")