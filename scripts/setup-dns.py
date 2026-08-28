#!/usr/bin/env python
"""
Safely set up DNS data for the boundary-layer turbulence modeling project.
Checks if data already exists before attempting expensive download.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import pypkg
sys.path.insert(0, str(Path(__file__).parent.parent))

from pypkg.jhtdb import read_stats

def setup_dns():
    """Ensure DNS data exists in data/jhtdb-transitional-bl/all-stats.h5"""
    data_dir = Path(__file__).parent.parent / "data" / "jhtdb-transitional-bl"
    h5_path = data_dir / "all-stats.h5"

    if h5_path.exists():
        print(f"DNS data already exists at {h5_path}")
        print("Skipping download.")
        return

    print(f"DNS data not found at {h5_path}")
    print("Attempting to download from JHTDB API...")
    print("NOTE: This requires JHTDB_TOKEN environment variable to be set.")

    # Ensure directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Call the read_stats function which handles caching
    read_stats()
    print(f"DNS data successfully set up at {h5_path}")

if __name__ == "__main__":
    setup_dns()
