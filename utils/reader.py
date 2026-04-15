import pandas as pd
import os
import re


def read_heater_excel(filepath):
    """
    Reads a heater test Excel file.
    Data is in Sheet2, with metadata in Sheet1.
    """
    df = pd.read_excel(filepath, sheet_name="Sheet2", engine="openpyxl")
    
    # Identify actual data columns (before empty columns)
    data_cols = []
    for col in df.columns:
        if col is None or str(col).startswith("Unnamed"):
            break
        data_cols.append(col)
    
    df = df[data_cols].copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = _convert_time_to_seconds(df)
    df = df.dropna(how="all")
    
    return df


def _convert_time_to_seconds(df):
    """
    Converts the time column to elapsed seconds.
    Correctly handles 12h (PM) and midnight crossovers.
    """
    time_col = df.columns[0]
    times = df[time_col].copy()
    
    # Convert time objects to total seconds since midnight
    total_seconds = []
    for t in times:
        if hasattr(t, 'hour'):
            secs = t.hour * 3600 + t.minute * 60 + t.second
            total_seconds.append(secs)
        else:
            total_seconds.append(None)
    
    # Calculate elapsed using consecutive differences
    # If the difference is negative, check if it's a 12h or 24h crossover
    elapsed = [0]
    for i in range(1, len(total_seconds)):
        if total_seconds[i] is not None and total_seconds[i-1] is not None:
            diff = total_seconds[i] - total_seconds[i-1]
            if diff < 0:
                # Try 12-hour correction first
                diff_12h = diff + 43200
                diff_24h = diff + 86400
                # If adding 12h gives a reasonable interval (< 60 sec), use that
                if 0 < diff_12h < 60:
                    diff = diff_12h
                else:
                    diff = diff_24h
            elapsed.append(elapsed[-1] + diff)
        else:
            elapsed.append(None)
    
    df[time_col] = elapsed
    df = df.rename(columns={time_col: "elapsed_seconds"})
    
    return df

def read_all_heaters(data_dir):
    """Reads all Excel files and combines them with a heater_id column."""
    xlsx_files = [f for f in os.listdir(data_dir) if f.endswith(".xlsx")]
    xlsx_files.sort()
    
    all_dfs = []
    
    for filename in xlsx_files:
        filepath = os.path.join(data_dir, filename)
        heater_id = filename.replace(".xlsx", "")
        
        print(f"  Reading: {filename}")
        df = read_heater_excel(filepath)
        df.insert(0, "heater_id", heater_id)
        
        print(f"    Rows: {len(df)} | Columns: {list(df.columns)}")
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def get_thermocouple_columns(df):
    """Returns only numbered TC columns (TC1, TC2, etc.)."""
    return [c for c in df.columns if re.match(r'^TC\d+$', c.strip())]


def get_ts_columns(df):
    """Returns TS columns."""
    return [c for c in df.columns if c.startswith("TS")]


def get_ot_columns(df):
    """Returns OT columns."""
    return [c for c in df.columns if c.startswith("OT")]