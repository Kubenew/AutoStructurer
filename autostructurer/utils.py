import os
import hashlib
import datetime

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sha1_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def format_ts(seconds: float) -> str:
    seconds = max(0, float(seconds))
    return str(datetime.timedelta(seconds=int(seconds)))
