import json
import argparse
from pathlib import Path
from time import time

from metrics_core import ZiWeiEvaluator


# ==========================
# Default Paths
# ==========================

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GOLD = ROOT / "data/final_data/test.jsonl"
DEFAULT_PRED = ROOT / "prediction/prediction_noSFT_noRAG_clean.jsonl"

TASK_NAME = "benchmark"
DEFAULT_PER_CASE = ROOT / f"evaluation_metric/results/{TASK_NAME}_metrics_per_cases.jsonl"
DEFAULT_SUMMARY = ROOT / f"evaluation_metric/results/{TASK_NAME}_metrics_summary.jsonl"


# ==========================
# Main
# ==========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Zi Wei Dou Shu model evaluator")

    parser.add_argument(
        "--gold", type=str, default=str(DEFAULT_GOLD),
        help="Path to gold/reference JSONL file"
    )
    parser.add_argument(
        "--pred", type=str, default=str(DEFAULT_PRED),
        help="Path to prediction JSONL file"
    )
    parser.add_argument(
        "--per_case_out", type=str, default=str(DEFAULT_PER_CASE),
        help="Output JSONL path for per-case metrics"
    )
    parser.add_argument(
        "--summary_out", type=str, default=str(DEFAULT_SUMMARY),
        help="Output JSON path for aggregated summary metrics"
    )

    args = parser.parse_args()

    print("Calculating evaluation metrics...")
    start = time()

    evaluator = ZiWeiEvaluator()
    per_case, summary = evaluator.evaluate_jsonl_files(
        gold_path=args.gold,
        pred_path=args.pred,
        per_case_output=args.per_case_out,
        summary_output=args.summary_out,
    )

    end = time()
    print(f"Computation time: {end - start:.2f} seconds")

    print("\nAggregated Metrics Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))