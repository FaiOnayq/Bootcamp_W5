import pandas as pd
from pathlib import Path

def load_csv(csv_path, encoding='utf-8'):
    """
    Load CSV file with proper error handling
    
    Args:
        csv_path: Path to CSV file
        encoding: File encoding (default: utf-8)
    
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
        return df
    except UnicodeDecodeError:
        # Try alternative encodings
        for enc in ['utf-8-sig', 'cp1256', 'iso-8859-6']:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                return df
            except:
                continue
        raise ValueError(f"Could not decode CSV file with common Arabic encodings")
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")

def validate_columns(df, required_columns):
    """
    Validate that required columns exist in DataFrame
    
    Args:
        df: pandas DataFrame
        required_columns: list of column names
    
    Returns:
        bool: True if all columns exist
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return True

def save_csv(df, output_path, encoding='utf-8'):
    """
    Save DataFrame to CSV with proper encoding
    
    Args:
        df: pandas DataFrame
        output_path: Output file path
        encoding: File encoding
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding=encoding)