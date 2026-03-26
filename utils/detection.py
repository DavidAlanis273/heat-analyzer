import pandas as pd
import numpy as np


def detect_zscore_anomalies(series, threshold=3.0):
    """
    Detecta anomalías usando Z-Score.
    Marca lecturas que están a más de 'threshold' desviaciones estándar del promedio.
    
    Args:
        series: pandas Series con valores de temperatura
        threshold: número de desviaciones estándar para considerar anomalía
    
    Returns:
        pandas Series de booleans (True = anomalía)
    """
    mean = series.mean()
    std = series.std()
    
    if std == 0 or pd.isna(std):
        return pd.Series(False, index=series.index)
    
    z_scores = (series - mean).abs() / std
    return z_scores > threshold


def detect_iqr_anomalies(series, multiplier=1.5):
    """
    Detecta anomalías usando IQR (Interquartile Range).
    Marca lecturas fuera de [Q1 - multiplier*IQR, Q3 + multiplier*IQR].
    
    Args:
        series: pandas Series con valores de temperatura
        multiplier: multiplicador del IQR para definir límites
    
    Returns:
        pandas Series de booleans (True = anomalía)
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    
    return (series < lower) | (series > upper)


def detect_isolation_forest_anomalies(df, feature_columns, contamination=0.05):
    """
    Detecta anomalías usando Isolation Forest (ML).
    Aprende patrones normales y marca lo que no encaja.
    
    Args:
        df: DataFrame con los datos
        feature_columns: lista de columnas a usar como features
        contamination: proporción esperada de anomalías (0.05 = 5%)
    
    Returns:
        pandas Series de booleans (True = anomalía)
    """
    from sklearn.ensemble import IsolationForest
    
    # Preparar datos (eliminar nulls para el modelo)
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
    # Isolation Forest devuelve -1 para anomalías, 1 para normal
    result[valid_mask] = predictions == -1
    
    return result


def detect_frozen_readings(series, window=10):
    """
    Detecta lecturas congeladas — cuando el sensor manda el mismo
    valor exacto durante 'window' lecturas consecutivas.
    
    Args:
        series: pandas Series con valores de temperatura
        window: número de lecturas idénticas consecutivas para marcar
    
    Returns:
        pandas Series de booleans (True = frozen)
    """
    result = pd.Series(False, index=series.index)
    
    # Calcular diferencia entre lecturas consecutivas
    diff = series.diff().abs()
    
    # Contar rachas de ceros (lecturas idénticas)
    is_same = diff == 0
    streak = is_same.groupby((~is_same).cumsum()).cumcount() + 1
    
    # Marcar donde la racha es >= window
    result = (is_same) & (streak >= window)
    
    return result


def run_all_detection(df, tc_column, settings):
    """
    Corre los 4 métodos de detección en una columna de termopar.
    
    Args:
        df: DataFrame de un solo heater
        tc_column: nombre de la columna del termopar
        settings: diccionario con parámetros de config
    
    Returns:
        DataFrame con resultados de detección
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
    
    # Isolation Forest (usa rolling features si existen)
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
    
    # Anomalía combinada: marcada por al menos 2 métodos
    methods = ['anomaly_zscore', 'anomaly_iqr', 'anomaly_iforest', 'anomaly_frozen']
    results['anomaly_count'] = results[methods].sum(axis=1)
    results['anomaly_consensus'] = results['anomaly_count'] >= 2
    
    return results