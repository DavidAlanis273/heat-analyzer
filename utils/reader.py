import pandas as pd
import os


def read_heater_excel(filepath):
    """
    Lee un archivo Excel de prueba de calefactor.
    Los datos están en Sheet2, con metadata en Sheet1.
    
    Args:
        filepath: ruta al archivo .xlsx
    
    Returns:
        DataFrame con los datos limpios de Sheet2
    """
    # Leer Sheet2 donde están los datos
    df = pd.read_excel(filepath, sheet_name="Sheet2", engine="openpyxl")
    
    # Identificar las columnas de datos reales (antes de las tablas de resumen)
    # Las tablas de resumen empiezan después de columnas con None en el header
    data_cols = []
    for col in df.columns:
        if col is None or str(col).startswith("Unnamed"):
            break
        data_cols.append(col)
    
    df = df[data_cols].copy()
    
    # Limpiar nombres de columnas (quitar espacios extra)
    df.columns = [str(c).strip() for c in df.columns]
    
    # Convertir la columna de tiempo a seconds elapsed
    df = _convert_time_to_seconds(df)
    
    # Eliminar filas completamente vacías
    df = df.dropna(how="all")
    
    return df


def _convert_time_to_seconds(df):
    """
    Convierte la columna Time (Sec) de hora del día a 
    segundos transcurridos desde el inicio de la prueba.
    """
    time_col = df.columns[0]  # primera columna es siempre el tiempo
    
    times = df[time_col].copy()
    
    # Convertir time objects a total seconds desde medianoche
    total_seconds = []
    for t in times:
        if hasattr(t, 'hour'):
            secs = t.hour * 3600 + t.minute * 60 + t.second
            total_seconds.append(secs)
        else:
            total_seconds.append(None)
    
    # Calcular elapsed seconds desde el primer reading
    start = total_seconds[0]
    elapsed = []
    for s in total_seconds:
        if s is not None and start is not None:
            diff = s - start
            if diff < 0:
                diff += 86400  # cruzó medianoche
            elapsed.append(diff)
        else:
            elapsed.append(None)
    
    df[time_col] = elapsed
    df = df.rename(columns={time_col: "elapsed_seconds"})
    
    return df


def read_all_heaters(data_dir):
    """
    Lee todos los archivos Excel en la carpeta data/ y los combina 
    en un solo DataFrame con una columna heater_id.
    
    Args:
        data_dir: ruta a la carpeta con los .xlsx
    
    Returns:
        DataFrame combinado con columna heater_id
    """
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
    """Devuelve la lista de columnas que son termopares (TC*)."""
    return [c for c in df.columns if c.startswith("TC") and not "PSwitch" in c]


def get_ts_columns(df):
    """Devuelve la lista de columnas TS."""
    return [c for c in df.columns if c.startswith("TS")]


def get_ot_columns(df):
    """Devuelve la lista de columnas OT."""
    return [c for c in df.columns if c.startswith("OT")]