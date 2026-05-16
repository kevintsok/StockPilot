#!/usr/bin/env python3
"""
Standalone dashboard runner - bypasses package __init__ to avoid torch import chain.
Usage: python run_dashboard.py
"""
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import the dashboard module directly
if __name__ == "__main__":
    from auto_select_stock.web.ops_dashboard import run
    run()