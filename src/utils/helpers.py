"""
Utility helper functions
"""

import uuid
from typing import Optional
from datetime import datetime


def generate_uuid() -> str:
    """
    Generate a new UUID v4
    
    Returns:
        String representation of UUID
    """
    return str(uuid.uuid4())


def is_valid_uuid(uuid_string: str) -> bool:
    """
    Check if a string is a valid UUID
    
    Args:
        uuid_string: String to validate
        
    Returns:
        True if valid UUID, False otherwise
    """
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, AttributeError):
        return False


def get_current_timestamp() -> str:
    """
    Get current UTC timestamp in ISO format
    
    Returns:
        ISO formatted timestamp string
    """
    return datetime.utcnow().isoformat()


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def extract_filename_from_uri(uri: str) -> Optional[str]:
    """
    Extract filename from URI
    
    Args:
        uri: The URI string
        
    Returns:
        Filename if found, None otherwise
    """
    try:
        return uri.split('/')[-1].split('?')[0]
    except (IndexError, AttributeError):
        return None
