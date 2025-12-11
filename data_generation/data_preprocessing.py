import json
import os
import random
from pathlib import Path
import argparse
import configs


def is_complete_interpretation(text: str) -> bool:
    """Heuristic completeness check."""
    if not text:
        return False
    if "整體判斷" not in text:
        return False
    if len(text) < 100:
        return False
    return True


def clean_record(record, stats):
    """Clean a single record, return cleaned dict or None if dropped."""
    # Check '解讀'
    if "解讀" not in record:
        stats["no_jiedu"] += 1
        return None

    text = record.get("解讀", "")
    if not isinstance(text, str):
        text = str(text)

    # Skip <think>
    if "<think>" in text:
        stats["think"] += 1
        return None

    # Skip incomplete
    if not is_complete_interpretation(text):
        stats["incomplete"] += 1
        return None

    # Remove personal info
    basic_info = record.get("命盤", {}).get("基本資料", {})
    if isinstance(basic_info, dict):
        basic_info.pop("出生日期", None)
        basic_info.pop("性別", None)

    return record


def clean_jsonl_file(input_path: Path, output_path: Path):
    """Clean one JSONL file and save output."""
    stats = {"kept": 0, "think": 0, "incomplete": 0, "no_jiedu": 0, "parse_error": 0}

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_error"] += 1
                continue

            cleaned = clean_record(record, stats)
            if cleaned is None:
                continue

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    print(f"[Cleaned] {input_path.name}")
    print(stats)
    return stats


def clean_all_in_dir(input_dir: Path, output_dir: Path):
    """Clean every jsonl file inside a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.jsonl"))
    print(f"Found {len(files)} files to clean.")
    for f in files:
        out_path = output_dir / f.name
        clean_jsonl_file(f, out_path)


def merge_jsonl_dir(input_dir: Path, output_path: Path):
    """Merge all files in a directory into one."""
    files = sorted(Path(input_dir).glob("*.jsonl"))
    print("Merging files:")
    total_lines = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for f in files:
            print(" -", f.name)
            with f.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
                        total_lines += 1

    print(f"Merged {len(files)} files ({total_lines} lines) → {output_path}")


def split_jsonl(input_path: Path, train_path: Path, val_path: Path, test_path: Path,
                train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split jsonl into train/val/test."""
    with input_path.open("r", encoding="utf-8") as fin:
        lines = [line.strip() for line in fin if line.strip()]

    random.seed(seed)
    random.shuffle(lines)

    n = len(lines)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = lines[:n_train]
    val = lines[n_train:n_train+n_val]
    test = lines[n_train+n_val:]

    train_path.write_text("\n".join(train), encoding="utf-8")
    val_path.write_text("\n".join(val), encoding="utf-8")
    test_path.write_text("\n".join(test), encoding="utf-8")

    print(f"Split → train:{len(train)} val:{len(val)} test:{len(test)}")


# -------------------------------
#           CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean, merge, and split interpretation JSONL files.")

    parser.add_argument("--clean", action="store_true", help="Clean all jsonl files in raw dir.")
    parser.add_argument("--merge", action="store_true", help="Merge cleaned files into one jsonl.")
    parser.add_argument("--split", action="store_true", help="Split merged jsonl into train/val/test.")

    args = parser.parse_args()

    RAW_DIR = configs.INTERPRETATIONS_DIR / "Qwen3-8B-FP8"
    CLEAN_DIR = configs.INTERPRETATIONS_CLEAN_DIR
    MERGED_PATH = configs.INTERPRETATIONS_MERGE_DIR / "input.jsonl"
    TRAIN = configs.INTERPRETATIONS_SPLIT_DIR / "train.jsonl"
    VAL = configs.INTERPRETATIONS_SPLIT_DIR / "val.jsonl"
    TEST = configs.INTERPRETATIONS_SPLIT_DIR / "test.jsonl"

    # 1) Clean (optional)
    if args.clean:
        clean_all_in_dir(RAW_DIR, CLEAN_DIR)

    # 2) Merge (optional)
    if args.merge:
        merge_jsonl_dir(CLEAN_DIR, MERGED_PATH)

    # 3) Split (optional)
    if args.split:
        split_jsonl(MERGED_PATH, TRAIN, VAL, TEST)
