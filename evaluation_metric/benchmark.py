import json
import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai


# ------------------------------------------------------------
#  Load Gemini API Key
# ------------------------------------------------------------
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
    print("Gemini API Key loaded.")
else:
    raise ValueError("❌ GEMINI_API_KEY is missing. Please set it in your .env file.")


# ------------------------------------------------------------
#  Select Gemini Model
# ------------------------------------------------------------
model = genai.GenerativeModel("gemini-2.5-flash")


# ------------------------------------------------------------
#  Input / Output Paths
# ------------------------------------------------------------
input_file = "/Users/ccy/Documents/CMU/Fall2025/10623 GenAI/ziwei-fortune-telling-llm/data/final_data/test.jsonl"
output_file = "prediction/benchmark.jsonl"


# ------------------------------------------------------------
#  Helper: Clean model output (remove ```json fences)
# ------------------------------------------------------------
def clean_json_output(text: str) -> str:
    """Remove Markdown fencing such as ```json ... ```."""
    text = text.strip()

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            cleaned = parts[1].replace("json", "", 1).strip()
            text = cleaned

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


# ------------------------------------------------------------
#  Build Prompt for Each Record
# ------------------------------------------------------------
def make_prompt(record: dict) -> str:
    template = '''
你是一位專精紫微斗數的命理師。
接下來會收到一組出生資料（生日、時辰_index、性別）。請你：

【任務】
1. 根據出生資料，自行排出完整的紫微斗數命盤，放在輸出 JSON 的 "命盤" 欄位中。
2. 根據你排出的命盤，只針對「事業運（含未來十年）」撰寫命理解讀文字，放在 "解讀" 欄位中。
3. 不需要解釋紫微斗數理論，只做命盤推演與解讀。

------------------------------------------------------------
【輸出格式 — 非常重要】
你只能輸出 **一個 JSON 物件**，結構如下：

{
  "命盤": { ... },
  "解讀": "..."
}

------------------------------------------------------------
【命盤規格要求】
"命盤" 欄位本身是一個物件，其內部有一層同名 "命盤"：

"命盤": {
  "命盤": {
    ...
  }
}

12 宮固定名稱：
"命宮"、"兄弟"、"夫妻"、"子女"、"财帛"、"疾厄"、
"迁移"、"仆役"、"官禄"、"田宅"、"福德"、"父母"

每宮格式如下：

"官禄": {
  "本命": {
    "主星": [...],
    "化曜": [...]
  },
  "大限": {
    "範圍": [起歲, 止歲],
    "天干": "X",
    "地支": "X"
  },
  "流年": {
    "對應年齡": [a, b, c]
  }
}

------------------------------------------------------------
【解讀格式】
"解讀": "（一）事業運\n　【本命分析】：...\n\n　【大限分析】：...\n\n　【整體判斷】：..."

- 使用繁體中文
- 專業但有溫度
- 每段 3–4 句，避免冗長

------------------------------------------------------------
【技術規則】
- 對宮對照必須遵守
- 空宮必須明講並借對宮
- 事業三方四正分析順序：官祿→財帛→遷移→命
- 年齡與大限需依 2025 作計算基準

------------------------------------------------------------
【以下是出生資料】
{{RECORD_JSON}}
'''

    return template.replace("{{RECORD_JSON}}", json.dumps(record, ensure_ascii=False))


# ------------------------------------------------------------
#  Main Generation Loop
# ------------------------------------------------------------
def main():
    # Load input JSONL
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    with open(output_file, "w", encoding="utf-8") as fout:

        for line in tqdm(lines, desc="Generating Zi Wei outputs", unit="case"):
            record_json = json.loads(line)
            record = record_json["出生資料"]

            # Build prompt
            prompt = make_prompt(record)

            # Call Gemini
            response = model.generate_content(prompt)
            raw = response.text.strip()

            # Clean markdown fencing
            cleaned = clean_json_output(raw)

            # Ensure valid JSON if possible
            try:
                obj = json.loads(cleaned)
                compact = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                # Fallback: return raw cleaned text
                compact = cleaned.replace("\n", "").strip()

            # Write single-line JSON
            fout.write(compact + "\n")

    print(f"Done! Output saved to: {output_file}")


if __name__ == "__main__":
    main()
