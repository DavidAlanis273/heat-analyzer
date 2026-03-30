# === Heater Performance Analyzer - Configuration ===

# Paths (todo dentro del repo)
DATA_DIR = "data"
OUTPUT_DIR = "outputs"

# Output files
CLEAN_DATA_FILE = "outputs/all_heaters_clean.csv"
FEATURES_FILE = "outputs/heaters_with_features.csv"
PROFILES_FILE = "outputs/heater_profiles.csv"
ANOMALIES_FILE = "outputs/anomalies.csv"
REPORT_FILE = "outputs/summary_report.csv"

# Sheet2 column mapping
TIME_COL = "Time (Sec)"
TC_PREFIX = "TC"
TS_PREFIX = "TS"
OT_PREFIX = "OT"

# Number of thermocouple columns to read (TC1 to TC25)
TC_COUNT = 25
# Number of TS and OT columns
TS_COUNT = 3
OT_COUNT = 3

# Feature engineering parameters
ROLLING_WINDOW = 30          # number of readings for rolling stats (30 x 5sec = 2.5 min)
READING_INTERVAL_SEC = 5     # seconds between readings

# Anomaly detection parameters
ZSCORE_THRESHOLD = 3.0       # standard deviations for Z-Score method
IQR_MULTIPLIER = 1.5         # multiplier for IQR method
ISOLATION_FOREST_CONTAMINATION = 0.05  # expected proportion of anomalies
FROZEN_WINDOW = 10           # consecutive identical readings to flag as frozen

# Set point tolerance for pass/fail
SETPOINT_TOLERANCE = 10.0    # +/- 10°C from set point to pass

# Ramp up target temperatures to measure time-to-reach
RAMP_TARGETS = [50, 100, 150, 200]  # °C