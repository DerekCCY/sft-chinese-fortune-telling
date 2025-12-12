# Ziwei-Fortune-Telling-LLM

*A complete pipeline for generating, interpreting, evaluating, and visualizing Zi Wei Dou Shu (紫微斗數) charts with LLMs.*

---

## 1. Environment Setup

```bash
conda env create -f environment.yaml -n ziwei
conda activate ziwei
```

---

## 2. Data Generation Pipeline

This project provides an automated pipeline to generate **birth data**, **Zi Wei Dou Shu natal charts**, and **career interpretation texts**.

### **2.1 Generate Random Birth Data (`gen_birth.py`)**

Randomly produces:

* Birth date (1965–2025)
* Birth time
* Gender
  Output → `data/births/`

---

### **2.2 Generate Natal Charts (`gen_charts.py`)**

Creates Ziwei natal charts using generated birth records.
Output → `data/charts/`

---

### **2.3 End-to-End Data Pipeline (`data_gen_pipeline.py`)**

#### **Run via Makefile**

```bash
make births      # generate births
make charts      # generate charts
make run         # generate both
```

Outputs:

* `data/births/`
* `data/charts/`

---

### **2.4 Interpretation Generation (`gen_interpretation.py`)**

Generates **career fortune interpretations** from natal charts using Qwen3-8B-FP8.

#### Prompt guideline

`interpretation_prompt_advanced_zh.txt`
Contains optimized prompt instructions for producing high-quality interpretation texts.

#### Run:

```bash
python -m data_generation.gen_interpretation
```

Output → `data/interpretation/Qwen3-8B-FP8/`

---

## 3. Data Preprocessing

### **3.1 Clean Interpretation Files**

```bash
python -m data_generation.data_preprocessing --clean
```

Cleaning rules:

* Remove entries containing `<think>`
* Remove entries missing `"解讀"`
* Remove incomplete interpretations (missing `"整體判斷"` etc.)
* Remove personal information under `"命盤.基本資料"` (birth date, gender)

---

### **3.2 Merge Cleaned Files**

```bash
python -m data_generation.data_preprocessing --merge
```

Output → `INTERPRETATIONS_MERGE_DIR/input.jsonl`

---

### **3.3 Split into Train / Val / Test**

```bash
python -m data_generation.data_preprocessing --split
```

---

## 4. Post-process Prediction Files

```bash
python -m data_generation.fix_jsonl --split
```

**fix_jsonl.py** performs:

* Parsing of records like `{"prediction": "<json_string>"}`
* Extract and validate `"命盤"` + `"解讀"`
* Remove broken or unparseable predictions
* Sync indices with the ground-truth file

⚠️ Update **absolute paths** in the script if needed.

---

## 🔮 5. Benchmarking (Gemini 2.5 Flash)

1. Add your API key into `.env`
2. This script:

   * Reads birth info from JSONL
   * Generates **full Ziwei chart**
   * Writes **career-focused interpretation**
   * Saves results into: `prediction/benchmark.jsonl`

Run:

```bash
python evaluation_metric/benchmark.py
```

---

## 6. Metric Evaluation

### **evaluation_metric/metrics_core.py**

The `ZiWeiEvaluator` computes both **structural chart metrics** and **interpretation metrics**.

Includes:

* Chart structural accuracy
* Jaccard similarity (main stars / transformed stars)
* Da-Xian range IoU
* Da-Xian Gan-Zhi accuracy
* Star–palace pair extraction from interpretation text
* Career topic F1
* Sentence-BERT cosine similarity
* BERTScore
* JSONL batch evaluation

### **Run evaluation**

```bash
python evaluation_metric/metrics_caculation.py
```

Outputs:

* Per-case metrics → `*_metrics_per_cases.jsonl`
* Summary metrics → `*_metrics_summary.jsonl`

---

## 7. Visualization

### **7.1 Basic Metric Visualization (`visualize_ziwei_metrics.py`)**

Generates:

* **Radar chart** (`radar_chart.jpg`)
* **Distribution histograms** (`distribution_chart.jpg`)
* **Career sub-topic bar chart** (`topic_chart.jpg`)

Run:

```bash
python visualization/visualize_ziwei_metrics.py
```

---

### **7.2 Model Comparison Visualization (`visualize_ziwei_metrics_compare.py`)**

Compares **four models**:

* Gemini-2.5-Flash
* Qwen3-4B (baseline)
* Ziwei (finetuned)
* Ziwei-RAG (RAG-enhanced)

Produces:

* **Comparison radar chart** (`radar_compare_4models.jpg`)

---

## 8. Training

```bash
cd training
python ft.py
```

---

## 9. Inference

### **9.1 Without RAG**

```bash
cd inference
python inference.py \
    --data_file data/test.jsonl \
    --output_file outputs/test_predictions.jsonl \
    --max_new_tokens 2048 \
    --use_lora
```

### **9.2 With RAG**

#### 1. Build RAG corpus

```bash
cd inference
python build_rag.py
```

#### 2. Build RAG index

```bash
python rag_build_index.py
```

#### 3. RAG inference

```bash
python rag_inference.py \
    --data_file data/test.jsonl \
    --knowledge_file rag/rag.jsonl \
    --output_file outputs/prediction_sft_rag.jsonl \
    --max_new_tokens 2048 \
    --rag_log_file outputs/sft_rag_log.jsonl
```

