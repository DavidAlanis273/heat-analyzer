# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install matplotlib --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 - Final Report
# MAGIC Genera visualizaciones y reporte detallado por archivo y por termopar.

# COMMAND ----------

import os
import sys
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

repo_root = '/Workspace/Users/david.alanis@watlow.com/heat-analyzer'
sys.path.insert(0, repo_root)

import utils.reader
importlib.reload(utils.reader)

from utils.reader import get_thermocouple_columns
from config.settings import (
    CLEAN_DATA_FILE, PROFILES_FILE, ANOMALIES_FILE,
    OUTPUT_DIR, REPORT_FILE
)

print(f"Repo root: {repo_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load data

# COMMAND ----------

df_clean = pd.read_csv(os.path.join(repo_root, CLEAN_DATA_FILE))
df_profiles = pd.read_csv(os.path.join(repo_root, PROFILES_FILE))
df_anomalies = pd.read_csv(os.path.join(repo_root, ANOMALIES_FILE))

tc_columns = get_thermocouple_columns(df_clean)

print(f"Clean data: {len(df_clean)} rows")
print(f"Profiles: {len(df_profiles)} thermocouple profiles")
print(f"Anomalies: {len(df_anomalies)} records ({int(df_anomalies['anomaly_consensus'].sum())} consensus)")
print(f"Thermocouples: {len(tc_columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Complete chart of ALL Thermocouples by file
# MAGIC Una gráfica por archivo mostrando todos los TC para ver la distribución de calor.

# COMMAND ----------

for heater_id in df_clean['heater_id'].unique():
    hdf = df_clean[df_clean['heater_id'] == heater_id]
    active_tcs = [tc for tc in tc_columns if hdf[tc].notna().any()]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    time_min = hdf['elapsed_seconds'] / 60
    
    for tc in active_tcs:
        ax.plot(time_min, hdf[tc], linewidth=0.6, alpha=0.7, label=tc)
    
    ax.set_title(f'All Thermocouples: {heater_id}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (minutes)', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontsize=11)
    ax.legend(fontsize=7, ncol=5, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = re.sub(r'[^\w]', '_', heater_id)
    plt.savefig(os.path.join(repo_root, OUTPUT_DIR, f'all_tcs_{safe_name}.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: all_tcs_{safe_name}.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Individual TC chart with anomalies marked
# MAGIC Para cada archivo, muestra los 4 TC más interesantes: el más caliente,
# MAGIC el más frío, el más inestable, y el que tiene más anomalías.

# COMMAND ----------

for heater_id in df_clean['heater_id'].unique():
    hdf = df_clean[df_clean['heater_id'] == heater_id]
    hp = df_profiles[df_profiles['heater_id'] == heater_id]
    h_anom = df_anomalies[(df_anomalies['heater_id'] == heater_id) & (df_anomalies['anomaly_consensus'] == True)]
    
    if len(hp) == 0:
        continue
    
    # Seleccionar los 4 TC más interesantes
    hottest = hp.loc[hp['temp_max'].idxmax(), 'thermocouple']
    coldest = hp.loc[hp['temp_max'].idxmin(), 'thermocouple']
    most_unstable = hp.loc[hp['stability'].idxmax(), 'thermocouple']
    
    if len(h_anom) > 0:
        most_anomalies = h_anom.groupby('thermocouple').size().idxmax()
    else:
        most_anomalies = hp.iloc[0]['thermocouple']
    
    tcs_to_plot = list(dict.fromkeys([hottest, coldest, most_unstable, most_anomalies]))[:4]
    
    # If there are less than 4, add more
    while len(tcs_to_plot) < 4:
        for tc in tc_columns:
            if tc not in tcs_to_plot and tc in hdf.columns and hdf[tc].notna().any():
                tcs_to_plot.append(tc)
                break
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Key Thermocouples: {heater_id}', fontsize=13, fontweight='bold')
    
    labels = {
        hottest: 'Hottest',
        coldest: 'Coldest',
        most_unstable: 'Most Unstable',
        most_anomalies: 'Most Anomalies'
    }
    
    for idx, tc in enumerate(tcs_to_plot):
        ax = axes[idx // 2][idx % 2]
        
        if tc in hdf.columns:
            time_min = hdf['elapsed_seconds'] / 60
            ax.plot(time_min, hdf[tc], color='#2E75B6', linewidth=0.5, alpha=0.8)
            
            # Marcar anomalías
            tc_anom = h_anom[h_anom['thermocouple'] == tc]
            if len(tc_anom) > 0:
                anom_time = tc_anom['elapsed_seconds'] / 60
                ax.scatter(anom_time, tc_anom['temperature'], color='red', s=10, alpha=0.6, zorder=5)
            
            label = labels.get(tc, '')
            tc_prof = hp[hp['thermocouple'] == tc]
            subtitle = f"{tc} ({label})" if label else tc
            if len(tc_prof) > 0:
                subtitle += f" | Max: {tc_prof.iloc[0]['temp_max']:.0f}°C | Anomalies: {len(tc_anom)}"
            
            ax.set_title(subtitle, fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Temp (°C)')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = re.sub(r'[^\w]', '_', heater_id)
    plt.savefig(os.path.join(repo_root, OUTPUT_DIR, f'key_tcs_{safe_name}.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: key_tcs_{safe_name}.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Heat Map - Max temperature by TC and file

# COMMAND ----------

heatmap_data = df_profiles.pivot_table(index='thermocouple', columns='heater_id', values='temp_max')
tc_order = sorted(heatmap_data.index, key=lambda x: int(re.search(r'\d+', x).group()))
heatmap_data = heatmap_data.loc[tc_order]

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')

ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index, fontsize=9)
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels([c[:25] for c in heatmap_data.columns], fontsize=8, rotation=20, ha='right')

for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        val = heatmap_data.values[i, j]
        if not np.isnan(val):
            color = 'white' if val > 150 else 'black'
            ax.text(j, i, f'{val:.0f}°', ha='center', va='center', fontsize=7, color=color)

ax.set_title('Maximum Temperature (°C) by Thermocouple and Test', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='Temperature (°C)')
plt.tight_layout()
plt.savefig(os.path.join(repo_root, OUTPUT_DIR, 'heatmap_max_temp.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Heatmap saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Anomalies chart - by method and by file

# COMMAND ----------

methods = ['anomaly_zscore', 'anomaly_iqr', 'anomaly_iforest', 'anomaly_frozen', 'anomaly_consensus']
method_labels = ['Z-Score', 'IQR', 'Isolation\nForest', 'Frozen', 'Consensus']

heaters = df_anomalies['heater_id'].unique()
x = np.arange(len(method_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
for i, heater_id in enumerate(heaters):
    hdf = df_anomalies[df_anomalies['heater_id'] == heater_id]
    counts = [int(hdf[m].sum()) for m in methods]
    short_name = heater_id[:25] + '...' if len(heater_id) > 25 else heater_id
    ax.bar(x + i * width, counts, width, label=short_name, alpha=0.85)

ax.set_ylabel('Number of Anomalies')
ax.set_title('Anomalies Detected by Method and Test', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(method_labels, fontsize=10)
ax.legend(fontsize=7)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(repo_root, OUTPUT_DIR, 'anomalies_by_method.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Final report by file - what engineers ned to see

# COMMAND ----------

print("=" * 75)
print("   HEATER PERFORMANCE ANALYSIS — FINAL REPORT")
print("=" * 75)

for heater_id in df_clean['heater_id'].unique():
    hdf = df_clean[df_clean['heater_id'] == heater_id]
    hp = df_profiles[df_profiles['heater_id'] == heater_id].copy()
    h_anom = df_anomalies[(df_anomalies['heater_id'] == heater_id) & (df_anomalies['anomaly_consensus'] == True)]
    
    duration_min = hdf['elapsed_seconds'].max() / 60
    duration_hrs = duration_min / 60
    active_tcs = [tc for tc in tc_columns if hdf[tc].notna().any()]
    
    print(f"\n{'─'*75}")
    print(f"  TEST: {heater_id}")
    print(f"{'─'*75}")
    print(f"  Duration: {duration_min:.1f} min ({duration_hrs:.1f} hrs) | Readings: {len(hdf)} | Active TCs: {len(active_tcs)}")
    
    # table of each TC
    print(f"\n  {'TC':<6} {'Start':>7} {'Final':>7} {'Max':>7} {'Avg':>7} {'Rate':>9} {'Stab':>6} {'Anom':>5}  Notes")
    print(f"  {'─'*85}")
    
    hp_sorted = hp.sort_values('thermocouple', key=lambda x: x.str.extract(r'(\d+)')[0].astype(int))
    
    for _, row in hp_sorted.iterrows():
        tc = row['thermocouple']
        tc_anom_count = len(h_anom[h_anom['thermocouple'] == tc])
        
        # generate automatic notes
        notes = []
        avg_max = hp['temp_max'].mean()
        
        if row['temp_max'] < avg_max * 0.5:
            notes.append("far from heat source")
        if row['temp_max'] - row['temp_min'] < 10:
            notes.append("barely moved")
        if row['stability'] > hp['stability'].quantile(0.9):
            notes.append("unstable")
        if row['overshoot'] > 10:
            notes.append(f"overshoot {row['overshoot']:.1f}°C")
        if tc_anom_count > 0:
            # Check if mostly during ramping
            tc_anom_detail = h_anom[h_anom['thermocouple'] == tc]
            if 'phase' in tc_anom_detail.columns:
                ramping_pct = len(tc_anom_detail[tc_anom_detail['phase'] == 'ramping']) / max(len(tc_anom_detail), 1)
                if ramping_pct > 0.8:
                    notes.append("anomalies during ramping (normal)")
                elif ramping_pct < 0.3:
                    notes.append("⚠ anomalies in steady state")
        
        notes_str = "; ".join(notes) if notes else "OK"
        
        print(f"  {tc:<6} {row['temp_initial']:>6.1f}° {row['temp_final']:>6.1f}° {row['temp_max']:>6.1f}° {row['temp_mean']:>6.1f}° {row['heating_rate_c_per_min']:>8.3f} {row['stability']:>6.2f} {tc_anom_count:>5}  {notes_str}")
    
    # General summary by file
    total_anom = len(h_anom)
    tcs_affected = h_anom['thermocouple'].nunique()
    
    print(f"\n  Summary:")
    print(f"    Total anomalies: {total_anom} across {tcs_affected} thermocouples")
    
    # specific recomendations 
    recommendations = []
    
    cold_tcs = hp[hp['temp_max'] < avg_max * 0.5]['thermocouple'].tolist()
    if cold_tcs:
        recommendations.append(f"TCs far from heat source ({', '.join(cold_tcs)}) — verify if positioning is intentional")
    
    unstable_tcs = hp[hp['stability'] > hp['stability'].quantile(0.9)]['thermocouple'].tolist()
    if unstable_tcs:
        recommendations.append(f"Unstable TCs ({', '.join(unstable_tcs)}) — review control loop or sensor connection")
    
    high_overshoot = hp[hp['overshoot'] > 10]
    if len(high_overshoot) > 0:
        os_tcs = high_overshoot['thermocouple'].tolist()
        recommendations.append(f"High overshoot ({', '.join(os_tcs)}) — consider tuning PID parameters")
    
    if not recommendations:
        recommendations.append("No significant issues detected — test looks clean")
    
    print(f"\n  Recommendations:")
    for r in recommendations:
        print(f"    → {r}")

print(f"\n{'='*75}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Save report as CSV

# COMMAND ----------

# Save report as CSV
report_rows = []
for heater_id in df_clean['heater_id'].unique():
    hdf = df_clean[df_clean['heater_id'] == heater_id]
    hp = df_profiles[df_profiles['heater_id'] == heater_id]
    h_anom = df_anomalies[(df_anomalies['heater_id'] == heater_id) & (df_anomalies['anomaly_consensus'] == True)]
    
    for _, row in hp.iterrows():
        tc = row['thermocouple']
        tc_anom_count = len(h_anom[h_anom['thermocouple'] == tc])
        report_rows.append({
            'heater_id': heater_id,
            'thermocouple': tc,
            'temp_initial': row['temp_initial'],
            'temp_final': row['temp_final'],
            'temp_min': row['temp_min'],
            'temp_max': row['temp_max'],
            'temp_mean': row['temp_mean'],
            'heating_rate': row['heating_rate_c_per_min'],
            'stability': row['stability'],
            'overshoot': row['overshoot'],
            'anomaly_count': tc_anom_count
        })

df_report = pd.DataFrame(report_rows)
report_path = os.path.join(repo_root, REPORT_FILE)
df_report.to_csv(report_path, index=False)
print(f"Report saved: {report_path} ({len(df_report)} rows)")

output_dir = os.path.join(repo_root, OUTPUT_DIR)
print(f"\nAll output files:")
for f in sorted(os.listdir(output_dir)):
    size = os.path.getsize(os.path.join(output_dir, f))
    print(f"  {f} ({size:,} bytes)")