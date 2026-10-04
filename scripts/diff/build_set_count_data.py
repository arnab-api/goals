#!/usr/bin/env python3
"""Run from any directory; see --help for bounded build options."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.diff.set_count_data import main

if __name__ == '__main__':
    main()
