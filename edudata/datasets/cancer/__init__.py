import pandas as pd
from pathlib import Path
import json
import platform


data_folder = Path(__file__).resolve().parent / "data"
dtypes_path = (data_folder / "dtypes.json")
csv_path = str(data_folder / "cancer.csv")


with dtypes_path.open('r') as f:
    dtypes = json.load(f)
columns = list(dtypes.keys())

if platform.system() == 'Darwin':
    df = pd.read_csv(csv_path, header=None, names=columns).astype(dtypes)
else:
    df = pd.read_csv(csv_path, header=None, names=columns, encoding='cp949').astype(dtypes)
