import sys
import os

# Ensure the deepdig root is on the path so `agent` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
