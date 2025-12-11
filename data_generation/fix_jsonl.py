import json
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, Any, List


# ------------------------------------------------------------
# 1. Extract inner JSON predictions and keep only valid entries
# ------------------------------------------------------------

def unwrap_predictions(in_path: str, out_path: str) -> List[int]:
    """
    Read a prediction JSONL file where each line looks like:
        {"prediction": "<json string>"}

    This function:
    - Parses the outer record
    - Parses the inner JSON string in `prediction`
    - Keeps only valid samples containing both "命盤" and "解讀"
    - Writes cleaned samples to out_path
    - Returns the indices of lines whose inner JSON failed to parse
    """

    cleaned_records: List[Dict[str, Any]] = []
    bad_indices: List[int] = []

    with open(in_path, "r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)

            # Only process samples where the model output "prediction" is a stringified JSON
            if isinstance(record.get("prediction"), str):
                pred_str = record["prediction"]

                # Try parsing the inner JSON (the model's output)
                try:
                    inner = json.loads(pred_str)
                except JSONDecodeError as e:
                    # Log badly formatted predictions for later reference
                    print(f"❌ Inner JSON failed at line {idx}")
                    print("   Error:", e)
                    print("   Sample:", pred_str[:200], "...")
                    bad_indices.append(idx)
                    continue

                # Keep only complete valid prediction structures
                if "命盤" in inner and "解讀" in inner:
                    cleaned_records.append(inner)
                else:
                    print("⚠️ Missing required keys:", inner)

    # Write all cleaned predictions to output file
    out_file = Path(out_path)
    with out_file.open("w", encoding="utf-8") as f:
        for rec in cleaned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(cleaned_records)} cleaned records to {out_file}")
    return bad_indices


# ------------------------------------------------------------
# 2. Drop corresponding bad lines from the label file
# ------------------------------------------------------------

def drop_bad_lines(
    label_in_path: str,
    label_out_path: str,
    bad_indices: List[int],
) -> None:
    """
    Given a list of bad line indices from the prediction file,
    remove the same lines from the label file (test.jsonl).

    This ensures predictions and labels remain aligned.
    """

    in_path = Path(label_in_path)
    out_path = Path(label_out_path)

    total = kept = dropped = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            total += 1

            # If this sample had a broken prediction, we drop the label too
            if idx in bad_indices:
                dropped += 1
                continue

            fout.write(line)
            kept += 1

    print(f"Input file : {in_path}")
    print(f"Output file: {out_path}")
    print(f"Total lines : {total}")
    print(f"Kept lines  : {kept}")
    print(f"Dropped lines: {dropped}")
    print(f"Bad indices  : {sorted(bad_indices)}")


# ------------------------------------------------------------
# 3. Main execution
# ------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: Extract valid predictions and record which lines failed
    bad_indices = unwrap_predictions(
        "/Users/ccy/Documents/CMU/Fall2025/10623 GenAI/ziwei-fortune-telling-llm/prediction/sft_norag_new.jsonl",
        "/Users/ccy/Documents/CMU/Fall2025/10623 GenAI/ziwei-fortune-telling-llm/prediction/sft_norag_new_clean.jsonl",
    )

    print("Bad prediction indices:", bad_indices)

    # Step 2: Remove the same bad lines from the label file to keep alignment
    drop_bad_lines(
        label_in_path="/Users/ccy/Documents/CMU/Fall2025/10623 GenAI/ziwei-fortune-telling-llm/data/final_data/test.jsonl",
        label_out_path="/Users/ccy/Documents/CMU/Fall2025/10623 GenAI/ziwei-fortune-telling-llm/data/final_data/test_temp.jsonl",
        bad_indices=bad_indices,
    )
