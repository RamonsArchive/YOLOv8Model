import os
import pandas as pd

def main():
    # 1. Locate the CSV
    curr_dir    = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(curr_dir, "runs", "train", "desk_objects", "results.csv")

    # 2. Read into a DataFrame
    df = pd.read_csv(results_path)
    df['fitness'] = (
    0.0 * df['metrics/precision(B)'] +
    0.0 * df['metrics/recall(B)']    +
    0.1 * df['metrics/mAP50(B)']     +
    0.9 * df['metrics/mAP50-95(B)']
)

    # 3. Choose which metric you care about most on the VAL set:
    #    - "metrics/mAP50(B)"   → mAP@.50
    #    - "metrics/mAP50-95(B)"→ COCO‐style mAP
    sort_col = "fitness"

    # 4. Sort descending, pick the best row
    df_sorted = df.sort_values(by=sort_col, ascending=False)
    best_row  = df_sorted.iloc[0]

    # 5. Report
    print(f"🏆 Best epoch by {sort_col}:")
    print(best_row.to_frame().T)  # as a single‐row table

if __name__ == "__main__":
    main()