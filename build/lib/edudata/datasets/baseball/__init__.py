import pandas as pd
from pathlib import Path
import json


data_folder = Path(__file__).resolve().parent / "data"
dtypes_path = (data_folder / "dtypes.json")
csv_path = str(data_folder / "baseball.csv")


with dtypes_path.open('rt', encoding='UTF8') as f:
    dtypes = json.load(f)
columns = list(dtypes.keys())
df = pd.read_csv(csv_path, header=None, names=columns).astype(dtypes)
