# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 05 - SQL Queries & Data Access
# MAGIC Registra los datos como tablas SQL para queries directos y dashboards.

# COMMAND ----------

import os
import pandas as pd

repo_root = '/Workspace/Users/david.alanis@watlow.com/heat-analyzer'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load CSVs and register as SQL views

# COMMAND ----------

df_clean = pd.read_csv(os.path.join(repo_root, "outputs/all_heaters_clean.csv"))
df_profiles = pd.read_csv(os.path.join(repo_root, "outputs/heater_profiles.csv"))
df_anomalies = pd.read_csv(os.path.join(repo_root, "outputs/anomalies.csv"))
df_report = pd.read_csv(os.path.join(repo_root, "outputs/summary_report.csv"))

sp_avg_path = os.path.join(repo_root, "outputs/setpoint_averages.csv")
sp_delta_path = os.path.join(repo_root, "outputs/setpoint_deltas.csv")

has_setpoints = os.path.exists(sp_avg_path)
if has_setpoints:
    df_sp_avg = pd.read_csv(sp_avg_path)
    df_sp_delta = pd.read_csv(sp_delta_path)

spark.createDataFrame(df_clean).createOrReplaceTempView("heater_readings")
spark.createDataFrame(df_profiles).createOrReplaceTempView("thermocouple_profiles")
spark.createDataFrame(df_anomalies).createOrReplaceTempView("anomalies")
spark.createDataFrame(df_report).createOrReplaceTempView("summary_report")

if has_setpoints:
    spark.createDataFrame(df_sp_avg).createOrReplaceTempView("setpoint_averages")
    spark.createDataFrame(df_sp_delta).createOrReplaceTempView("setpoint_deltas")

print("SQL views registered successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 1: General summary by test

# COMMAND ----------

display(spark.sql("""
SELECT 
  heater_id AS test_name,
  COUNT(*) AS total_readings,
  ROUND(MAX(elapsed_seconds) / 60, 1) AS duration_min,
  ROUND(MAX(elapsed_seconds) / 3600, 1) AS duration_hrs
FROM heater_readings
GROUP BY heater_id
ORDER BY total_readings DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 2: Complete profile of each thermocouple test
 
# COMMAND ----------

display(spark.sql("""
SELECT 
  heater_id AS test_name,
  thermocouple AS tc,
  temp_initial AS start_temp,
  temp_final AS end_temp,
  temp_min,
  temp_max,
  ROUND(temp_mean, 1) AS avg_temp,
  ROUND(heating_rate_c_per_min, 3) AS heating_rate,
  ROUND(stability, 3) AS stability,
  ROUND(overshoot, 1) AS overshoot
FROM thermocouple_profiles
ORDER BY heater_id, temp_max DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 3: Thermocouples with an overshoot greater than 5°C

# COMMAND ----------

display(spark.sql("""
SELECT 
  heater_id AS test_name,
  thermocouple AS tc,
  ROUND(overshoot, 1) AS overshoot,
  temp_max AS max_temp,
  temp_final AS final_temp
FROM thermocouple_profiles
WHERE overshoot > 5
ORDER BY overshoot DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 4: Most inestable thermocouples (top 10)

# COMMAND ----------

display(spark.sql("""
SELECT 
  heater_id AS test_name,
  thermocouple AS tc,
  ROUND(stability, 3) AS stability,
  ROUND(temp_mean, 1) AS avg_temp,
  temp_max AS max_temp
FROM thermocouple_profiles
ORDER BY stability DESC
LIMIT 10
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 5: Cold Thermocouples (less than 50% of the average maximum temperature)

# COMMAND ----------

display(spark.sql("""
WITH avg_max AS (
  SELECT heater_id, AVG(temp_max) AS avg_max_temp
  FROM thermocouple_profiles
  GROUP BY heater_id
)
SELECT 
  p.heater_id AS test_name,
  p.thermocouple AS tc,
  p.temp_max,
  ROUND(a.avg_max_temp, 1) AS avg_max_all_tcs,
  ROUND(p.temp_max / a.avg_max_temp * 100, 1) AS pct_of_avg
FROM thermocouple_profiles p
JOIN avg_max a ON p.heater_id = a.heater_id
WHERE p.temp_max < a.avg_max_temp * 0.5
ORDER BY pct_of_avg ASC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 6: Anomailes summary by test and thermocouple

# COMMAND ----------

display(spark.sql("""
SELECT 
  heater_id AS test_name,
  thermocouple AS tc,
  COUNT(*) AS total_flagged,
  SUM(CASE WHEN anomaly_consensus = true THEN 1 ELSE 0 END) AS consensus_anomalies,
  SUM(CASE WHEN anomaly_zscore = true THEN 1 ELSE 0 END) AS zscore_anomalies,
  SUM(CASE WHEN anomaly_iqr = true THEN 1 ELSE 0 END) AS iqr_anomalies,
  SUM(CASE WHEN anomaly_iforest = true THEN 1 ELSE 0 END) AS iforest_anomalies,
  ROUND(AVG(temperature), 1) AS avg_anomaly_temp
FROM anomalies
WHERE anomaly_consensus = true
GROUP BY heater_id, thermocouple
ORDER BY consensus_anomalies DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 7: Severe anomalies (3+ methods)

# COMMAND ----------

display(spark.sql("""
SELECT 
  heater_id AS test_name,
  thermocouple AS tc,
  elapsed_seconds,
  ROUND(temperature, 1) AS temperature,
  anomaly_count AS methods_detected,
  phase
FROM anomalies
WHERE anomaly_count >= 3
ORDER BY anomaly_count DESC, heater_id, elapsed_seconds
LIMIT 20
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 8: Anomalies in steady state (the ones that matter)

# COMMAND ----------

display(spark.sql("""
SELECT 
  heater_id AS test_name,
  thermocouple AS tc,
  COUNT(*) AS steady_state_anomalies,
  ROUND(MIN(temperature), 1) AS min_temp,
  ROUND(MAX(temperature), 1) AS max_temp,
  ROUND(AVG(temperature), 1) AS avg_temp
FROM anomalies
WHERE anomaly_consensus = true AND phase = 'steady_state'
GROUP BY heater_id, thermocouple
ORDER BY steady_state_anomalies DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 9: Set point averages

# COMMAND ----------

display(spark.sql("SELECT * FROM setpoint_averages ORDER BY heater_id, set_point"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 10: Deltas per set point

# COMMAND ----------

display(spark.sql("SELECT * FROM setpoint_deltas ORDER BY heater_id, set_point"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 11: A specific TC in all tests (replace TC7 with whichever one you want)

# COMMAND ----------

display(spark.sql("""
SELECT 
  heater_id AS test_name,
  thermocouple AS tc,
  temp_initial AS start_temp,
  temp_max AS max_temp,
  ROUND(heating_rate_c_per_min, 3) AS heating_rate,
  ROUND(stability, 3) AS stability,
  ROUND(overshoot, 1) AS overshoot
FROM thermocouple_profiles
WHERE thermocouple = 'TC7'
ORDER BY heater_id
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 12: Top 5 fastest Thermocouples bases on testings

# COMMAND ----------

display(spark.sql("""
SELECT * FROM (
  SELECT 
    heater_id AS test_name,
    thermocouple AS tc,
    ROUND(heating_rate_c_per_min, 3) AS heating_rate,
    temp_max AS max_temp,
    RANK() OVER (PARTITION BY heater_id ORDER BY heating_rate_c_per_min DESC) AS rank
  FROM thermocouple_profiles
)
WHERE rank <= 5
ORDER BY test_name, rank
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tablas disponibles
# MAGIC - `heater_readings` — datos crudos limpios
# MAGIC - `thermocouple_profiles` — perfiles por TC
# MAGIC - `anomalies` — anomalías detectadas
# MAGIC - `summary_report` — reporte final
# MAGIC - `setpoint_averages` — promedios por set point
# MAGIC - `setpoint_deltas` — deltas por set point