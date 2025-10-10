# scripts/check_athena.py
import os, sys, pandas as pd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.features.clean import load_from_athena, basic_date_parse
from src.utils.text import torch_device_name

print("[Info] Device:", torch_device_name())
df = load_from_athena()  # uses ATHENA_SQL from .env by default
df = basic_date_parse(df)
print(df.head())
print("Rows:", len(df))