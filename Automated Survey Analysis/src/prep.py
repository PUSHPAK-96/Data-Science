
import re
import pandas as pd

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<[^>]+>", " ", s)  # strip HTML
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s

def load_csv(path_or_buf) -> pd.DataFrame:
    df = pd.read_csv(path_or_buf)
    df.columns = [c.strip().lower() for c in df.columns]
    if "free_text" not in df.columns:
        raise ValueError("CSV must contain a 'free_text' column")
    df["free_text"] = df["free_text"].fillna("").map(clean_text)
    return df
