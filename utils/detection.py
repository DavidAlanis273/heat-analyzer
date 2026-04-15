import pandas as pd
import numpy as np


def detect_zscore_anomalies(series, threshold=3.0):
    """
    Detects anomalies using Z-Score.
    Flags readings that are more than 'threshold' standard deviations from the mean.
    
    Args:
        series: pandas Series with temperature values
        threshold: number of standard deviations to consider an anomaly
    
    Returns:
        pandas Series of booleans (True = anomaly)
    """
    mean = series.mean()
    std = series.std()
    
    if std == 0 or pd.isna(std):
        return pd.Series(False, index=series.index)
    
    z_scores = (series - mean).abs() / std
    return z_scores > threshold


def detect_iqr_anomalies(series, multiplier=1.5):
    """
    Detects anomalies using IQR (Interquartile Range).
    Flags readings outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR].
    
    Args:
        series: pandas Series with temperature values
        multiplier: IQR multiplier to define bounds
    
    Returns:
        pandas Series of booleans (True = anomaly)
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    
    return (series < lower) | (series > upper)


def detect_isolation_forest_anomalies(df, feature_columns, contamination=0.05):
    """
    Detects anomalies using Isolation Forest (ML).
    Learns normal patterns and flags what doesn't fit.
    
    Args:
        df: DataFrame with the data
        feature_columns: list of columns to use as features
        contamination: expected proportion of anomalies (0.05 = 5%)
    
    Returns:
        pandas Series of booleans (True = anomaly)
    """
    from sklearn.ensemble import IsolationForest
    
    # Prepare data (remove nulls for the model)
    features = df[feature_columns].copy()
    valid_mask = features.notna().all(axis=1)
    
    result = pd.Series(False, index=df.index)
    
    if valid_mask.sum() < 10:
        return result
    
    X = features[valid_mask].values
    
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    predictions = model.fit_predict(X)
    # Isolation Forest returns -1 for anomalies, 1 for normal
    result[valid_mask] = predictions == -1
    
    return result


def detect_frozen_readings(series, window=10):
    """
    Detects frozen readings — when the sensor sends the same
    exact value for 'window' consecutive readings.
    
    Args:
        series: pandas Series with temperature values
        window: number of identical consecutive readings to flag
    
    Returns:
        pandas Series of booleans (True = frozen)
    """
    result = pd.Series(False, index=series.index)
    
    # Calculate difference between consecutive readings
    diff = series.diff().abs()
    
    # Count streaks of zeros (identical readings)
    is_same = diff == 0
    streak = is_same.groupby((~is_same).cumsum()).cumcount() + 1
    
    # Flag where the streak is >= window
    result = (is_same) & (streak >= window)
    
    return result


def run_all_detection(df, tc_column, settings):
    """
    Runs all 4 detection methods on a thermocouple column.
    
    Args:
        df: DataFrame for a single heater
        tc_column: name of the thermocouple column
        settings: dictionary with configuration parameters
    
    Returns:
        DataFrame with detection results
    """
    series = df[tc_column].dropna()
    valid_idx = series.index
    
    results = pd.DataFrame(index=valid_idx)
    results['elapsed_seconds'] = df.loc[valid_idx, 'elapsed_seconds']
    results['temperature'] = series
    results['thermocouple'] = tc_column
    
    # Z-Score
    results['anomaly_zscore'] = detect_zscore_anomalies(
        series, threshold=settings['zscore_threshold']
    )
    
    # IQR
    results['anomaly_iqr'] = detect_iqr_anomalies(
        series, multiplier=settings['iqr_multiplier']
    )
    
    # Frozen
    results['anomaly_frozen'] = detect_frozen_readings(
        series, window=settings['frozen_window']
    )
    
    # Isolation Forest (uses rolling features if they exist)
    feature_cols = [tc_column]
    roc_col = f"{tc_column}_rate_of_change"
    ravg_col = f"{tc_column}_rolling_avg"
    rstd_col = f"{tc_column}_rolling_std"
    
    for col in [roc_col, ravg_col, rstd_col]:
        if col in df.columns:
            feature_cols.append(col)
    
    results['anomaly_iforest'] = detect_isolation_forest_anomalies(
        df.loc[valid_idx], feature_cols, 
        contamination=settings['iforest_contamination']
    )
    
    # Combined anomaly: flagged by at least 2 methods
    methods = ['anomaly_zscore', 'anomaly_iqr', 'anomaly_iforest', 'anomaly_frozen']
    results['anomaly_count'] = results[methods].sum(axis=1)
    results['anomaly_consensus'] = results['anomaly_count'] >= 2
    
    return results