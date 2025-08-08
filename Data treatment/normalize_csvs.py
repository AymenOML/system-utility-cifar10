# data_interpretation/normalize_csvs.py
import argparse
import os
import sys
import pandas as pd
import numpy as np
from typing import List

# Columns we should never normalize (add more if needed)
DEFAULT_EXCLUDE_BY_NAME = {
    "rank", "client_rank", "round", "timestamp", "client_id", "server_id",
    "node", "host", "epoch", "batch", "tag"
}

def is_integer_series(s: pd.Series) -> bool:
    # True if dtype is integer-like (including nullable Int64)
    return pd.api.types.is_integer_dtype(s.dtype)

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s.dtype)

def minmax_normalize(col: pd.Series) -> pd.Series:
    # Ignore NaNs for min/max; keep NaNs as NaN in the result
    col_min = col.min(skipna=True)
    col_max = col.max(skipna=True)
    # Constant or all-NaN column -> return zeros (preserving NaNs)
    if pd.isna(col_min) or pd.isna(col_max) or np.isclose(col_max, col_min):
        return col.where(col.isna(), 0.0).astype(float)
    return (col - col_min) / (col_max - col_min)

def normalize_file(path: str, out_dir: str = None,
                   exclude_by_name: set = None,
                   dry_run: bool = False) -> str:
    exclude_by_name = exclude_by_name or set()
    df = pd.read_csv(path)

    # Decide which columns to normalize:
    # - must be numeric
    # - not integer dtype (to keep identifiers untouched)
    # - not in explicit exclude list by name (case-insensitive)
    lower_name_map = {c: c.lower() for c in df.columns}
    exclude_cols = {c for c in df.columns if lower_name_map[c] in exclude_by_name}

    cols_to_normalize = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        s = df[c]
        if is_numeric_series(s) and not is_integer_series(s):
            cols_to_normalize.append(c)

    # Apply normalization
    df_norm = df.copy()
    for c in cols_to_normalize:
        df_norm[c] = minmax_normalize(df_norm[c])

    # Output path
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    out_name = f"{name}_normalized{ext}"
    out_dir = out_dir or os.path.dirname(path) or "."
    out_path = os.path.join(out_dir, out_name)

    if dry_run:
        print(f"[DRY-RUN] Would write: {out_path}")
        return out_path

    # Ensure output dir exists
    os.makedirs(out_dir, exist_ok=True)
    df_norm.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    return out_path

def normalize_many(paths: List[str], out_dir: str = None,
                   extra_excludes: List[str] = None,
                   dry_run: bool = False) -> None:
    exclude = set(x.lower() for x in DEFAULT_EXCLUDE_BY_NAME)
    if extra_excludes:
        exclude |= set(x.lower() for x in extra_excludes)
    for p in paths:
        normalize_file(p, out_dir=out_dir, exclude_by_name=exclude, dry_run=dry_run)

def main():
    parser = argparse.ArgumentParser(
        description="Min–max normalize numeric (float) columns in CSVs while preserving integer ID-like fields."
    )
    parser.add_argument("files", nargs="+", help="CSV file(s) to normalize")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: alongside each file)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Additional column names to exclude from normalization (case-insensitive)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    args = parser.parse_args()

    try:
        normalize_many(args.files, out_dir=args.out_dir, extra_excludes=args.exclude, dry_run=args.dry_run)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
