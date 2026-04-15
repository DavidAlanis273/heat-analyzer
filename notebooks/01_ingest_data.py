# Databricks notebook source


# COMMAND ----------

# MAGIC %pip install openpyxl --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 - Data Ingestion
# MAGIC Lee los archivos Excel de pruebas de calefactores, estandariza el formato
# MAGIC y genera un CSV limpio combinado.

# COMMAND ----------

import os
import sys
import pandas as pd

repo_root = '/Workspace/Users/david.alanis@watlow.com/heat-analyzer'
sys.path.insert(0, repo_root)

from utils.reader import read_all_heaters, get_thermocouple_columns, get_ts_columns, get_ot_columns
from config.settings import DATA_DIR, OUTPUT_DIR, CLEAN_DATA_FILE

print(f"Repo root: {repo_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Read all excel files

# COMMAND ----------

data_path = os.path.join(repo_root, DATA_DIR)

xlsx_files = [f for f in os.listdir(data_path) if f.endswith(".xlsx")]
print(f"Archivos encontrados: {len(xlsx_files)}")
for f in sorted(xlsx_files):
    print(f"  - {f}")

# COMMAND ----------

df = read_all_heaters(data_path)
print(f"\nDataFrame combinado: {df.shape[0]} rows x {df.shape[1]} columns")

# COMMAND ----------

import os
base = '/Workspace/Users/david.alanis@watlow.com'
print(f"Base exists: {os.path.exists(base)}")
if os.path.exists(base):
    for f in os.listdir(base):
        print(f"  {f}")
else:
    print("Trying without /Workspace...")
    base2 = '/Users/david.alanis@watlow.com'
    print(f"Base2 exists: {os.path.exists(base2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Explore data

# COMMAND ----------


print("Columnas disponibles:")
for col in df.columns:
    non_null = df[col].notna().sum()
    print(f"  {col}: {non_null}/{len(df)} values ({df[col].dtype})")

# COMMAND ----------

# Summanry by heater
for heater_id in df['heater_id'].unique():
    hdf = df[df['heater_id'] == heater_id]
    tc_cols = get_thermocouple_columns(hdf)
    duration_sec = hdf['elapsed_seconds'].max()
    duration_min = duration_sec / 60 if duration_sec else 0
    print(f"\n{heater_id}:")
    print(f"  Readings: {len(hdf)}")
    print(f"  Duration: {duration_min:.1f} minutes")
    print(f"  Thermocouples: {len(tc_cols)} active")
    print(f"  Temp range: {hdf[tc_cols].min().min():.1f}°C to {hdf[tc_cols].max().max():.1f}°C")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: clean data

# COMMAND ----------

# Count nulls
null_count_before = df.isnull().sum().sum()
print(f"Total null values: {null_count_before}")

# Identify empty columns
for heater_id in df['heater_id'].unique():
    hdf = df[df['heater_id'] == heater_id]
    empty_cols = [c for c in hdf.columns if hdf[c].isnull().all()]
    if empty_cols:
        print(f"  {heater_id}: empty columns -> {empty_cols}")

# No eliminamos columnas vacías del DataFrame combinado porque 
# un TC puede estar activo en un archivo y no en otro
print(f"\nDataFrame shape after cleaning: {df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: save clean CSV

# COMMAND ----------

output_path = os.path.join(repo_root, OUTPUT_DIR)
os.makedirs(output_path, exist_ok=True)

clean_file = os.path.join(repo_root, CLEAN_DATA_FILE)
df.to_csv(clean_file, index=False)

print(f"Archivo guardado: {clean_file}")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final verification

# COMMAND ----------

df_verify = pd.read_csv(clean_file)
print(f"Verificación - Rows: {len(df_verify)} | Columns: {len(df_verify.columns)}")
print(f"Heaters: {df_verify['heater_id'].unique()}")
df_verify.head()
