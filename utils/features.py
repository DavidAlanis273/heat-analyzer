import pandas as pd
import numpy as np


def add_rolling_features(df, column, window=30):
    """
    Adds rolling average, rolling std, and rate of change to a column.
    
    Args:
        df: DataFrame with the data
        column: name of the temperature column
        window: window size for rolling stats
    
    Returns:
        DataFrame with the new columns added
    """
    df[f"{column}_rolling_avg"] = df[column].rolling(window=window, min_periods=1).mean()
    df[f"{column}_rolling_std"] = df[column].rolling(window=window, min_periods=1).std()
    df[f"{column}_rate_of_change"] = df[column].diff()
    
    return df


def compute_thermocouple_profile(df, tc_column, elapsed_col="elapsed_seconds", interval_sec=5):
    """
    Computes the full profile of a thermocouple.
    
    Args:
        df: DataFrame for a single heater
        tc_column: name of the thermocouple column (e.g. 'TC1')
        elapsed_col: time column in seconds
        interval_sec: seconds between readings
    
    Returns:
        Dictionary with all thermocouple metrics
    """
    series = df[tc_column].dropna()
    
    if len(series) == 0:
        return None
    
    elapsed = df[elapsed_col]
    duration_min = (elapsed.max() - elapsed.min()) / 60
    
    # Basic temperatures
    temp_initial = series.iloc[0]
    temp_final = series.iloc[-1]
    temp_min = series.min()
    temp_max = series.max()
    temp_mean = series.mean()
    temp_std = series.std()
    
    # Heating rate (°C per minute)
    if duration_min > 0:
        heating_rate = (temp_max - temp_initial) / duration_min
    else:
        heating_rate = 0.0
    
    # Overshoot: difference between the max and the final value
    overshoot = temp_max - temp_final
    
    # Stability: std of the last 10% of readings (steady state)
    last_10_pct = series.tail(max(1, len(series) // 10))
    stability = last_10_pct.std()
    
    # Total range
    temp_range = temp_max - temp_min
    
    # Rate of change stats
    roc = series.diff().dropna()
    max_roc = roc.max()  # maximum rise between readings
    min_roc = roc.min()  # maximum drop between readings
    avg_roc = roc.mean()
    
    return {
        "thermocouple": tc_column,
        "temp_initial": round(temp_initial, 2),
        "temp_final": round(temp_final, 2),
        "temp_min": round(temp_min, 2),
        "temp_max": round(temp_max, 2),
        "temp_mean": round(temp_mean, 2),
        "temp_std": round(temp_std, 2),
        "temp_range": round(temp_range, 2),
        "duration_min": round(duration_min, 2),
        "heating_rate_c_per_min": round(heating_rate, 4),
        "overshoot": round(overshoot, 2),
        "stability": round(stability, 4) if stability == stability else 0.0,
        "max_rate_of_change": round(max_roc, 4),
        "min_rate_of_change": round(min_roc, 4),
        "avg_rate_of_change": round(avg_roc, 4),
        "num_readings": len(series)
    }


def compute_heater_profiles(df, tc_columns, heater_id):
    """
    Computes the profile of all thermocouples for a heater.
    
    Args:
        df: DataFrame for a single heater
        tc_columns: list of thermocouple columns
        heater_id: heater identifier
    
    Returns:
        DataFrame with one row per thermocouple
    """
    profiles = []
    
    for tc in tc_columns:
        if tc in df.columns and df[tc].notna().any():
            profile = compute_thermocouple_profile(df, tc)
            if profile:
                profile["heater_id"] = heater_id
                profiles.append(profile)
    
    return pd.DataFrame(profiles)

def detect_set_points(df, tc_columns, window=60, roc_threshold=0.05):
    """
    Automatically detects steady-state phases (set points)
    by looking for periods where the average rate of change is near zero.
    """
    active_tcs = [tc for tc in tc_columns if tc in df.columns and df[tc].notna().any()]
    avg_temp = df[active_tcs].mean(axis=1)
    
    roc = avg_temp.diff().abs().rolling(window=window, min_periods=1).mean()
    stable = roc < roc_threshold
    groups = (stable != stable.shift()).cumsum()
    
    set_points = []
    for group_id in groups[stable].unique():
        mask = (groups == group_id) & stable
        if mask.sum() < window:
            continue
        
        group_temps = avg_temp[mask]
        group_elapsed = df.loc[mask, 'elapsed_seconds']
        
        sp_value = round(group_temps.mean())
        
        if set_points and abs(sp_value - set_points[-1]['set_point_approx']) < 10:
            continue
        
        set_points.append({
            'set_point_approx': sp_value,
            'start_seconds': group_elapsed.iloc[0],
            'end_seconds': group_elapsed.iloc[-1],
            'duration_min': (group_elapsed.iloc[-1] - group_elapsed.iloc[0]) / 60,
            'start_idx': mask.idxmax(),
            'end_idx': mask[::-1].idxmax(),
            'num_readings': mask.sum()
        })
    
    return set_points


def compute_setpoint_averages(df, tc_columns, set_points):
    """
    Computes the average of each TC during each set point phase.
    """
    rows = []
    for sp in set_points:
        mask = (df['elapsed_seconds'] >= sp['start_seconds']) & \
               (df['elapsed_seconds'] <= sp['end_seconds'])
        sp_data = df[mask]
        
        row = {'set_point': sp['set_point_approx']}
        for tc in tc_columns:
            if tc in sp_data.columns and sp_data[tc].notna().any():
                row[tc] = round(sp_data[tc].mean(), 2)
            else:
                row[tc] = None
        rows.append(row)
    
    return pd.DataFrame(rows)


def compute_setpoint_deltas(avg_table):
    """
    Computes the difference (delta) between each TC's average and the set point.
    """
    delta_rows = []
    for _, row in avg_table.iterrows():
        sp = row['set_point']
        delta_row = {'set_point': sp}
        for col in avg_table.columns:
            if col != 'set_point' and row[col] is not None:
                delta_row[col] = round(row[col] - sp, 2)
        delta_rows.append(delta_row)
    
    return pd.DataFrame(delta_rows)


def compute_pass_fail(avg_table, delta_table, tolerance=10.0):
    """
    Evaluates whether each thermocouple passes or fails against the specification.
    Passes if within +/- tolerance of the set point.
    
    Returns:
        DataFrame with Pass/Fail per TC per set point
    """
    results = []
    
    for _, row in delta_table.iterrows():
        sp = row['set_point']
        result_row = {'set_point': sp}
        pass_count = 0
        fail_count = 0
        
        for col in delta_table.columns:
            if col == 'set_point' or col == 'heater_id':
                continue
            val = row.get(col)
            if val is not None and not pd.isna(val):
                if abs(val) <= tolerance:
                    result_row[col] = 'PASS'
                    pass_count += 1
                else:
                    result_row[col] = f'FAIL ({val:+.1f}°)'
                    fail_count += 1
        
        result_row['total_pass'] = pass_count
        result_row['total_fail'] = fail_count
        result_row['pass_rate'] = round(pass_count / max(pass_count + fail_count, 1) * 100, 1)
        results.append(result_row)
    
    return pd.DataFrame(results)


def compute_ramp_up_time(df, tc_columns, targets, elapsed_col='elapsed_seconds'):
    """
    Computes how long each thermocouple takes to reach each target temperature.
    
    Args:
        df: DataFrame for a single heater
        tc_columns: list of TC columns
        targets: list of target temperatures [50, 100, 150, 200]
        elapsed_col: time column
    
    Returns:
        DataFrame with time in minutes to reach each target
    """
    results = []
    
    for tc in tc_columns:
        if tc not in df.columns or df[tc].isna().all():
            continue
        
        series = df[tc].dropna()
        elapsed = df.loc[series.index, elapsed_col]
        start_temp = series.iloc[0]
        start_time = elapsed.iloc[0]
        
        row = {'thermocouple': tc, 'start_temp': round(start_temp, 1)}
        
        for target in targets:
            # Only compute if the target is above the initial temperature
            if target <= start_temp:
                row[f'time_to_{target}C_min'] = 'already above'
                continue
            
            # Find the first reading that reaches or exceeds the target
            reached = series[series >= target]
            
            if len(reached) > 0:
                first_idx = reached.index[0]
                time_to_reach = (elapsed.loc[first_idx] - start_time) / 60
                row[f'time_to_{target}C_min'] = round(time_to_reach, 1)
            else:
                row[f'time_to_{target}C_min'] = 'never reached'
        
        results.append(row)
    
    return pd.DataFrame(results)