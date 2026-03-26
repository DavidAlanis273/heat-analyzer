# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install matplotlib --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 - Reporte Final
# MAGIC Genera visualizaciones y resumen ejecutivo del análisis completo.

# COMMAND ----------

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

repo_root = '/Workspace/Users/david.alanis@watlow.com/heat-analyzer'
sys.path.insert(0, repo_root)

from config.settings import (
    CLEAN_DATA_FILE, PROFILES_FILE, ANOMALIES_FILE, 
    OUTPUT_DIR, REPORT_FILE
)

print(f"Repo root: {repo_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1: Cargar todos los datos

# COMMAND ----------

df_clean = pd.read_csv(os.path.join(repo_root, CLEAN_DATA_FILE))
df_profiles = pd.read_csv(os.path.join(repo_root, PROFILES_FILE))
df_anomalies = pd.read_csv(os.path.join(repo_root, ANOMALIES_FILE))

print(f"Clean data: {len(df_clean)} rows")
print(f"Profiles: {len(df_profiles)} thermocouple profiles")
print(f"Anomalies: {len(df_anomalies)} anomaly records ({int(df_anomalies['anomaly_consensus'].sum())} consensus)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2: Gráfica de temperatura por heater con anomalías marcadas

# COMMAND ----------

# Seleccionar 4 termopares representativos para graficar
tcs_to_plot = ['TC1', 'TC7', 'TC18', 'TC24']

for heater_id in df_clean['heater_id'].unique():
    hdf = df_clean[df_clean['heater_id'] == heater_id]
    h_anom = df_anomalies[(df_anomalies['heater_id'] == heater_id) & (df_anomalies['anomaly_consensus'] == True)]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Temperature Profiles: {heater_id}', fontsize=14, fontweight='bold')
    
    for idx, tc in enumerate(tcs_to_plot):
        ax = axes[idx // 2][idx % 2]
        
        if tc in hdf.columns:
            # Serie de tiempo completa
            time_min = hdf['elapsed_seconds'] / 60
            ax.plot(time_min, hdf[tc], color='#2E75B6', linewidth=0.5, alpha=0.8, label='Normal')
            
            # Marcar anomalías
            tc_anom = h_anom[h_anom['thermocouple'] == tc]
            if len(tc_anom) > 0:
                anom_time = tc_anom['elapsed_seconds'] / 60
                ax.scatter(anom_time, tc_anom['temperature'], color='red', s=8, alpha=0.6, label=f'Anomalies ({len(tc_anom)})', zorder=5)
            
            ax.set_title(f'{tc}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Temperature (°C)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(repo_root, OUTPUT_DIR, f'plot_{heater_id.replace(" ", "_")}.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Plot saved for: {heater_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3: Mapa de calor — temperatura máxima por termopar y heater

# COMMAND ----------

# Pivot: rows = termopares, columns = heaters, values = temp_max
heatmap_data = df_profiles.pivot_table(
    index='thermocouple', 
    columns='heater_id', 
    values='temp_max'
)

# Ordenar termopares numéricamente
import re
tc_order = sorted(heatmap_data.index, key=lambda x: int(re.search(r'\d+', x).group()))
heatmap_data = heatmap_data.loc[tc_order]

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')

ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index, fontsize=9)
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels(heatmap_data.columns, fontsize=8, rotation=20, ha='right')

# Agregar valores en cada celda
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        val = heatmap_data.values[i, j]
        if not np.isnan(val):
            color = 'white' if val > 150 else 'black'
            ax.text(j, i, f'{val:.0f}°C', ha='center', va='center', fontsize=7, color=color)

ax.set_title('Maximum Temperature by Thermocouple and Heater Test', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='Temperature (°C)')
plt.tight_layout()
plt.savefig(os.path.join(repo_root, OUTPUT_DIR, 'heatmap_max_temp.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Heatmap saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4: Comparativa de anomalías por método y heater

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
    short_name = heater_id[:20] + '...' if len(heater_id) > 20 else heater_id
    ax.bar(x + i * width, counts, width, label=short_name, alpha=0.85)

ax.set_ylabel('Number of Anomalies')
ax.set_title('Anomalies Detected by Method and Heater', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(method_labels, fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(repo_root, OUTPUT_DIR, 'anomalies_by_method.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Anomaly comparison chart saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5: Top 10 termopares con más anomalías

# COMMAND ----------

tc_ranking = df_anomalies[df_anomalies['anomaly_consensus'] == True].groupby('thermocouple').size()
tc_ranking = tc_ranking.sort_values(ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#E74C3C' if v > tc_ranking.median() else '#F39C12' for v in tc_ranking.values]
tc_ranking.plot(kind='barh', ax=ax, color=colors)

ax.set_xlabel('Number of Consensus Anomalies')
ax.set_title('Top 10 Thermocouples with Most Anomalies', fontsize=13, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)

for i, v in enumerate(tc_ranking.values):
    ax.text(v + 5, i, str(v), va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(repo_root, OUTPUT_DIR, 'top_anomaly_thermocouples.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Top anomaly chart saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6: Generar tabla de reporte final

# COMMAND ----------

# Resumen por heater
report_rows = []

for heater_id in df_clean['heater_id'].unique():
    hdf = df_clean[df_clean['heater_id'] == heater_id]
    h_profiles = df_profiles[df_profiles['heater_id'] == heater_id]
    h_anom = df_anomalies[(df_anomalies['heater_id'] == heater_id) & (df_anomalies['anomaly_consensus'] == True)]
    
    duration_min = hdf['elapsed_seconds'].max() / 60
    num_readings = len(hdf)
    num_tcs = len(h_profiles)
    avg_max_temp = h_profiles['temp_max'].mean()
    avg_heating_rate = h_profiles['heating_rate_c_per_min'].mean()
    avg_stability = h_profiles['stability'].mean()
    total_anomalies = len(h_anom)
    tcs_with_anomalies = h_anom['thermocouple'].nunique()
    
    # Recomendación automática
    recommendations = []
    if avg_stability > 3:
        recommendations.append("High variability in steady state — review temperature control settings")
    if total_anomalies > 1000:
        recommendations.append("High anomaly count — investigate sensor placement and calibration")
    if tcs_with_anomalies > 20:
        recommendations.append("Anomalies across most thermocouples — possible system-wide issue")
    
    # Termopares con comportamiento inusual
    low_temp_tcs = h_profiles[h_profiles['temp_max'] < h_profiles['temp_max'].mean() * 0.5]
    if len(low_temp_tcs) > 0:
        tc_list = ', '.join(low_temp_tcs['thermocouple'].tolist())
        recommendations.append(f"Low temperature TCs ({tc_list}) — may be far from heat source or disconnected")
    
    if not recommendations:
        recommendations.append("No significant issues detected")
    
    report_rows.append({
        'heater_id': heater_id,
        'duration_min': round(duration_min, 1),
        'num_readings': num_readings,
        'active_thermocouples': num_tcs,
        'avg_max_temp_c': round(avg_max_temp, 1),
        'avg_heating_rate_c_per_min': round(avg_heating_rate, 4),
        'avg_stability': round(avg_stability, 4),
        'total_consensus_anomalies': total_anomalies,
        'tcs_with_anomalies': tcs_with_anomalies,
        'recommendations': ' | '.join(recommendations)
    })

df_report = pd.DataFrame(report_rows)

# COMMAND ----------

# Mostrar reporte
print("=" * 70)
print("   HEATER PERFORMANCE ANALYSIS - FINAL REPORT")
print("=" * 70)

for _, row in df_report.iterrows():
    print(f"\n  {row['heater_id']}")
    print(f"  {'─' * 50}")
    print(f"    Duration:          {row['duration_min']} minutes")
    print(f"    Readings:          {row['num_readings']}")
    print(f"    Active TCs:        {row['active_thermocouples']}")
    print(f"    Avg Max Temp:      {row['avg_max_temp_c']}°C")
    print(f"    Avg Heating Rate:  {row['avg_heating_rate_c_per_min']}°C/min")
    print(f"    Avg Stability:     {row['avg_stability']}")
    print(f"    Anomalies:         {row['total_consensus_anomalies']} across {row['tcs_with_anomalies']} TCs")
    print(f"    Recommendations:   {row['recommendations']}")

print(f"\n{'=' * 70}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 7: Guardar reporte

# COMMAND ----------

report_path = os.path.join(repo_root, REPORT_FILE)
df_report.to_csv(report_path, index=False)
print(f"Report saved: {report_path}")

# Lista de todos los archivos generados
output_dir = os.path.join(repo_root, OUTPUT_DIR)
print(f"\nAll output files:")
for f in sorted(os.listdir(output_dir)):
    size = os.path.getsize(os.path.join(output_dir, f))
    print(f"  {f} ({size:,} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen del proyecto completo

# COMMAND ----------

total_readings = df_clean.shape[0]
total_tcs = 25
total_anomalies = int(df_anomalies['anomaly_consensus'].sum())

print("=" * 70)
print("   HEATER PERFORMANCE ANALYZER - PROJECT SUMMARY")
print("=" * 70)
print(f"   Pipeline: 4 notebooks (Ingest → Features → Anomaly → Report)")
print(f"   Heater tests analyzed:   {df_clean['heater_id'].nunique()}")
print(f"   Total readings:          {total_readings:,}")
print(f"   Thermocouples per test:  {total_tcs}")
print(f"   Features calculated:     {len(df_profiles.columns) - 2} per thermocouple")
print(f"   Detection methods:       Z-Score, IQR, Isolation Forest, Frozen")
print(f"   Consensus anomalies:     {total_anomalies:,} ({total_anomalies/(total_readings*total_tcs)*100:.2f}%)")
print(f"   Visualizations:          4 charts generated")
print(f"   Automated recommendations generated per heater")
print(f"\n   What took hours of manual Excel review now takes seconds.")
print("=" * 70)