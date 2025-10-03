def safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float, handling quoted strings and nulls"""
    if value is None or value == 'null' or value == '':
        return default
    try:
        # Remove quotes if present
        if isinstance(value, str):
            value = value.strip('"\'')
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default: int = 0) -> int:
    """Safely convert a value to int, handling quoted strings and nulls"""
    if value is None or value == 'null' or value == '':
        return default
    try:
        # Remove quotes if present
        if isinstance(value, str):
            value = value.strip('"\'')
        return int(float(value))  # Convert to float first to handle decimal strings
    except (ValueError, TypeError):
        return default