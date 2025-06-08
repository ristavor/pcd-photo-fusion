"""
Utility functions for the calibration GUI.
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def validate_file_path(file_path: str, file_type: str) -> bool:
    """
    Validate if a file exists and has the correct extension.
    
    Args:
        file_path: Path to the file
        file_type: Type of file ('image', 'pcd', 'calib')
    
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not file_path:
        return False
    
    path = Path(file_path)
    if not path.exists():
        return False
    
    # Check file extensions
    if file_type == 'image':
        return path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    elif file_type == 'pcd':
        return path.suffix.lower() == '.pcd'
    elif file_type == 'calib':
        return path.suffix.lower() == '.txt'
    
    return False

def load_calibration_results(file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load calibration results from a JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Tuple of (R_matrix, T_vector) or (None, None) if loading fails
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        R_matrix = np.array(data['R_matrix'])
        T_vector = np.array(data['T_vector'])
        
        return R_matrix, T_vector
    except Exception:
        return None, None

def format_matrix_for_display(matrix: np.ndarray, precision: int = 6) -> str:
    """
    Format a numpy matrix for display in the GUI.
    
    Args:
        matrix: Numpy array to format
        precision: Number of decimal places to show
    
    Returns:
        Formatted string representation of the matrix
    """
    with np.printoptions(precision=precision, suppress=True):
        return str(matrix) 