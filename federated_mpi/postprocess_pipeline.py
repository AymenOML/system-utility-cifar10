import os, sys, traceback
# Make the parent directory importable first
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# now imports:
from normalize_csvs import normalize_many
import normalize_confusion_csvs as ncc
import data_labelling as dl


def run_postprocessing(base_data_dir="Data", fail_job_on_error=False):
    """
    1) Normalize CSVs (clients_stats, clients_system)
    2) Normalize + tidy confusion matrices
    3) Label dataset (equal-importance, per-round)
    Outputs are written under base_data_dir.
    """
    try:

        logs_dir = os.path.join(base_data_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # 1) Normalize the two main CSVs
        stats_csv = os.path.join(logs_dir, "clients_stats.csv")
        sys_csv   = os.path.join(logs_dir, "clients_system.csv")
        normalize_many([stats_csv, sys_csv], out_dir=None, extra_excludes=None, dry_run=False)
        stats_norm = os.path.join(logs_dir, "clients_stats_normalized.csv")
        sys_norm   = os.path.join(logs_dir, "clients_system_normalized.csv")

        # 2) Normalize + tidy all confusion matrices from Data/logs/confusion_csv/**
        ncc.CONFUSION_DIR = os.path.join(logs_dir, "confusion_csv")
        ncc.OUT_DIR = os.path.join(ncc.CONFUSION_DIR, "normalized")
        ncc.TIDY_CSV_PATH = os.path.join(ncc.OUT_DIR, "confusion_matrices_tidy.csv")
        ncc.main()  # writes normalized matrices + the tidy CSV

        conf_tidy = ncc.TIDY_CSV_PATH

        # 3) Labelling (write final dataset + journal in base_data_dir)
        out_csv = os.path.join(base_data_dir, "clients_labeled_equal_importance_with_confusion.csv")
        journal = os.path.join(base_data_dir, "selection_journal.txt")

        all_clients_stats, system_metrics, confusion_tidy = dl.load_datasets(
            stats_norm, sys_norm, conf_tidy
        )
        confusion_agg = dl.aggregate_confusion(confusion_tidy)
        merged_all = dl.merge_datasets(all_clients_stats, system_metrics, confusion_agg)

        classified_df, thresholds, cutoff, numeric_cols = dl.classify_equal_importance(
            merged_all, per_round=True
        )
        dl.save_dataset(classified_df, out_csv)
        dl.write_selection_journal(classified_df, numeric_cols, journal)

        print("[postprocess] Done.")
        return True

    except Exception as e:
        print("[postprocess] ERROR:", e)
        traceback.print_exc()
        if fail_job_on_error:
            raise
        return False
