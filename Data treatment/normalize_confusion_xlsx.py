import os
import sys
import glob
import numpy as np
import pandas as pd

CONFUSION_DIR = "confusion_matrices"  # folder containing all client_x.xlsx files
OUT_DIR = os.path.join(CONFUSION_DIR, "normalized")  # output folder for normalized files
TIDY_CSV_PATH = os.path.join(OUT_DIR, "confusion_matrices_tidy.csv")


def minmax_normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Min–max normalize numeric cells of a DataFrame to [0, 1], preserving non-numeric."""
    df_norm = df.copy()
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    vmin = np.nanmin(numeric_df.values)
    vmax = np.nanmax(numeric_df.values)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmax, vmin):
        return df.where(~numeric_df.notna(), 0.0)

    scaled = (numeric_df - vmin) / (vmax - vmin)
    return df.where(~numeric_df.notna(), scaled)


def load_sheet(xls_path: str, sheet_name: str) -> pd.DataFrame:
    """Load one sheet from Excel, preserving structure."""
    df = pd.read_excel(xls_path, sheet_name=sheet_name, engine="openpyxl")
    first_col = df.columns[0]
    if isinstance(first_col, str) and first_col.lower().startswith("unnamed"):
        df = df.set_index(df.columns[0])
    return df


def normalize_xlsx(in_path: str, out_dir: str) -> str:
    """Normalize all sheets in an Excel workbook and save to out_dir."""
    xfile = pd.ExcelFile(in_path, engine="openpyxl")
    sheets = xfile.sheet_names

    base = os.path.basename(in_path)
    name, ext = os.path.splitext(base)
    out_name = f"{name}_normalized{ext}"
    out_path = os.path.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
        for s in sheets:
            df = load_sheet(in_path, s)
            df_norm = minmax_normalize_frame(df)
            df_norm.to_excel(writer, sheet_name=s)

    print(f"Wrote normalized: {out_path}")
    return out_path


def tidy_sheet(df: pd.DataFrame, client_file: str, sheet_name: str) -> pd.DataFrame:
    """Convert a confusion matrix sheet into long-form rows."""
    if df.index.name is None:
        df.index.name = "true_label"
    if df.columns.name is None:
        df.columns.name = "pred_label"

    long_df = df.reset_index().melt(id_vars=[df.index.name],
                                    var_name="pred_label",
                                    value_name="value")
    long_df.insert(0, "client_file", os.path.basename(client_file))
    long_df.insert(1, "sheet", sheet_name)
    return long_df


def main():
    excel_files = sorted(glob.glob(os.path.join(CONFUSION_DIR, "*.xlsx")))
    if not excel_files:
        print(f"No .xlsx files found in {CONFUSION_DIR}", file=sys.stderr)
        sys.exit(1)

    all_long = []
    for path in excel_files:
        try:
            out_xlsx = normalize_xlsx(path, OUT_DIR)
        except Exception as e:
            print(f"[WARN] Skipped {path}: {e}", file=sys.stderr)
            continue

        # Build tidy CSV data
        xfile = pd.ExcelFile(out_xlsx, engine="openpyxl")
        for s in xfile.sheet_names:
            df = load_sheet(out_xlsx, s)
            long_df = tidy_sheet(df, client_file=out_xlsx, sheet_name=s)
            all_long.append(long_df)

    if all_long:
        tidy = pd.concat(all_long, ignore_index=True)
        tidy.to_csv(TIDY_CSV_PATH, index=False)
        print(f"Wrote tidy CSV: {TIDY_CSV_PATH}")


if __name__ == "__main__":
    main()
