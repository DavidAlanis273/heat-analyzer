import pandas as pd
import numpy as np


def add_rolling_features(df, column, window=30):
    """
    Agrega rolling average, rolling std y rate of change a una columna.
    
    Args:
        df: DataFrame con los datos
        column: nombre de la columna de temperatura
        window: tamaño de la ventana para rolling stats
    
    Returns:
        DataFrame con las columnas nuevas agregadas
    """
    df[f"{column}_rolling_avg"] = df[column].rolling(window=window, min_periods=1).mean()
    df[f"{column}_rolling_std"] = df[column].rolling(window=window, min_periods=1).std()
    df[f"{column}_rate_of_change"] = df[column].diff()
    
    return df


def compute_thermocouple_profile(df, tc_column, elapsed_col="elapsed_seconds", interval_sec=5):
    """
    Calcula el perfil completo de un termopar.
    
    Args:
        df: DataFrame de un solo heater
        tc_column: nombre de la columna del termopar (ej: 'TC1')
        elapsed_col: columna de tiempo en segundos
        interval_sec: segundos entre lecturas
    
    Returns:
        Diccionario con todas las métricas del termopar
    """
    series = df[tc_column].dropna()
    
    if len(series) == 0:
        return None
    
    elapsed = df[elapsed_col]
    duration_min = (elapsed.max() - elapsed.min()) / 60
    
    # Temperaturas básicas
    temp_initial = series.iloc[0]
    temp_final = series.iloc[-1]
    temp_min = series.min()
    temp_max = series.max()
    temp_mean = series.mean()
    temp_std = series.std()
    
    # Tasa de calentamiento (°C por minuto)
    if duration_min > 0:
        heating_rate = (temp_max - temp_initial) / duration_min
    else:
        heating_rate = 0.0
    
    # Overshoot: diferencia entre el máximo y el valor final
    overshoot = temp_max - temp_final
    
    # Estabilidad: std de la última 10% de lecturas (estado estable)
    last_10_pct = series.tail(max(1, len(series) // 10))
    stability = last_10_pct.std()
    
    # Rango total
    temp_range = temp_max - temp_min
    
    # Rate of change stats
    roc = series.diff().dropna()
    max_roc = roc.max()  # máxima subida entre lecturas
    min_roc = roc.min()  # máxima bajada entre lecturas
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
    Calcula el perfil de todos los termopares de un heater.
    
    Args:
        df: DataFrame de un solo heater
        tc_columns: lista de columnas de termopares
        heater_id: identificador del heater
    
    Returns:
        DataFrame con una fila por termopar
    """
    profiles = []
    
    for tc in tc_columns:
        if tc in df.columns and df[tc].notna().any():
            profile = compute_thermocouple_profile(df, tc)
            if profile:
                profile["heater_id"] = heater_id
                profiles.append(profile)
    
    return pd.DataFrame(profiles)