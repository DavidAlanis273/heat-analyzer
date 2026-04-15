#heater Analyzer-Config

# Pathd
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

# Number of thermocouple columns to read
TC_COUNT = 25
# Number of TS and OT columns
TS_COUNT = 3
OT_COUNT = 3

# Feature engineering parameters
ROLLING_WINDOW = 30          
READING_INTERVAL_SEC = 5     

# Anomaly detection parameters
ZSCORE_THRESHOLD = 3.0       
IQR_MULTIPLIER = 1.5         
ISOLATION_FOREST_CONTAMINATION = 0.05  
FROZEN_WINDOW = 10           

# Set point tolerance for pass/fai
SETPOINT_TOLERANCE = 10.0   

# Ramp up target temperatures to measure time-to-reach
RAMP_TARGETS = [50, 100, 150, 200]  # °C