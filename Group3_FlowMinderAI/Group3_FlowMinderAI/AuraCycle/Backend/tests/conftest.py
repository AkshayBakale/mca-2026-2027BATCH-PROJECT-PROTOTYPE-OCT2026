"""
conftest.py - Pytest Configuration & Path Setup
Ensures all modular folders, engines, and the backend are cleanly importable in any CI environment.
"""

import sys
import os

_test_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dirs = [
    _test_dir,
    os.path.dirname(_test_dir),
    os.path.dirname(os.path.dirname(_test_dir))
]

for base in _parent_dirs:
    for candidate in [
        base,
        os.path.join(base, "03_backend_api"),
        os.path.join(base, "02_ml_engine"),
        os.path.join(base, "01_data_pipeline"),
        os.path.join(base, "Folder_3_Backend_And_Team_Leader"),
        os.path.join(base, "Folder_2_ML_Engine"),
        os.path.join(base, "Folder_1_Data_Pipeline"),
        os.path.join(base, "auracycle_core")
    ]:
        if os.path.exists(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
            