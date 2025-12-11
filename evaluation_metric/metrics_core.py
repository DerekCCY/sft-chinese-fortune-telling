import json
from typing import Dict, Any, Set, List, Tuple, Optional

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore


class ZiWeiEvaluator:
    """
    Comprehensive evaluator for Zi Wei Dou Shu chart prediction and interpretation.

    This evaluator includes:
    - Structural chart scoring (stars, transformed stars, da-xian ranges, gan-zhi)
    - Star–palace pair consistency extracted from interpretation text
    - Topic-based interpretation coverage
    - Text similarity scores (cosine similarity, BERTScore)
    - Batch evaluation for aligned gold/pred JSONL files
    """

    # -----------------------------
    # Initialization
    # -----------------------------
    def __init__(
        self,
        sentence_model_name: str = "all-MiniLM-L6-v2",
        bert_lang: str = "zh",
    ):
        self.cos_model = SentenceTransformer(sentence_model_name)
        self.bert_lang = bert_lang

        # Palace name variations for text matching
        self.PALACE_VARIANTS = {
            "命宫": ["命宫", "命宮"],
            "兄弟": ["兄弟"],
            "夫妻": ["夫妻"],
            "子女": ["子女"],
            "财帛": ["财帛", "財帛"],
            "疾厄": ["疾厄"],
            "迁移": ["迁移", "遷移"],
            "仆役": ["仆役", "僕役"],
            "官禄": ["官禄", "官祿"],
            "田宅": ["田宅"],
            "福德": ["福德"],
            "父母": ["父母"],
        }

        # Topic keywords relevant for career interpretation
        self.TOPIC_KEYWORDS = {
            "career_role": {
                "work_context": ["事業", "工作", "職場", "職涯", "公司", "職業"],
                "position": ["角色", "職位", "崗位"],
                "management": ["主管", "領導", "管理"],
                "teamwork": ["團隊", "合作", "協調", "溝通"],
                "project_responsibility": ["專案", "職責"],
            },
            "career_wealth": {
                "financial_status": ["財帛", "財務", "財運", "財富"],
                "income": ["收入", "薪水", "報酬", "穩定收入"],
                "profit": ["獲利"],
                "investment_risk": ["投資", "風險", "報酬率"],
                "assets": ["資產", "資金"],
            },
            "career_location": {
                "movement": ["遷移"],
                "travel": ["外出"],
                "non_local": ["異地", "外地"],
                "international": ["國外", "出國"],
                "environment_change": ["外在環境", "環境變化"],
                "relocation": ["搬家", "移居", "移民"],
                "real_estate": ["不動產", "房產", "家庭相關事業", "田宅"],
                "local": ["在地", "本地"],
            },
        }

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _safe_get(d: Dict, *keys, default=None):
        """Safely retrieve nested dictionary values."""
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @staticmethod
    def _normalize_star_tokens(raw) -> Set[str]:
        """
        Normalize star token formats into a set of star names.
        Handles:
        - ["紫微", "天機"]
        - [{"星名": "天樞", ...}]
        - {"天樞": "財帛宮"}
        """
        if raw is None:
            return set()

        if not isinstance(raw, (list, tuple, set)):
            raw = [raw]

        result = []

        for x in raw:
            if isinstance(x, str):
                result.append(x)
                continue

            if isinstance(x, dict):
                if isinstance(x.get("星名"), str):
                    result.append(x["星名"])
                    continue
                if isinstance(x.get("名稱"), str):
                    result.append(x["名稱"])
                    continue

                if len(x) == 1:
                    k, v = next(iter(x.items()))
                    if isinstance(k, str):
                        result.append(k)
                        continue
                    if isinstance(v, str):
                        result.append(v)
                        continue

        return set(result)

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        """Jaccard similarity of two sets."""
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    @staticmethod
    def normalize_text(s: str) -> str:
        """Normalize common traditional–simplified variants."""
        mapping = {
            "宮": "宫", "陰": "阴", "陽": "阳", "財": "财",
            "遷": "迁", "祿": "禄", "權": "权", "貪": "贪", "門": "门"
        }
        for k, v in mapping.items():
            s = s.replace(k, v)
        return s

    # -----------------------------
    # Structural scores
    # -----------------------------
    def chart_accuracy_exact(self, gold_chart, pred_chart) -> float:
        """Strict exact match of main stars + transformed stars per palace."""
        correct = 0
        total = 0

        for palace, ginfo in gold_chart.items():
            gold_main = self._normalize_star_tokens(
                self._safe_get(ginfo, "本命", "主星", default=[])
            )
            gold_trans = self._normalize_star_tokens(
                self._safe_get(ginfo, "本命", "化曜", default=[])
            )

            pinfo = pred_chart.get(palace, {})
            pred_main = self._normalize_star_tokens(
                self._safe_get(pinfo, "本命", "主星", default=[])
            )
            pred_trans = self._normalize_star_tokens(
                self._safe_get(pinfo, "本命", "化曜", default=[])
            )

            total += 1
            if gold_main == pred_main and gold_trans == pred_trans:
                correct += 1

        return correct / total if total else 0.0

    def chart_star_jaccard(self, gold_chart, pred_chart):
        """Compute Jaccard similarities for main stars and transformed stars."""
        main_scores = []
        trans_scores = []

        for palace, ginfo in gold_chart.items():
            gold_main = self._normalize_star_tokens(
                self._safe_get(ginfo, "本命", "主星", default=[])
            )
            gold_trans = self._normalize_star_tokens(
                self._safe_get(ginfo, "本命", "化曜", default=[])
            )

            pinfo = pred_chart.get(palace, {})
            pred_main = self._normalize_star_tokens(
                self._safe_get(pinfo, "本命", "主星", default=[])
            )
            pred_trans = self._normalize_star_tokens(
                self._safe_get(pinfo, "本命", "化曜", default=[])
            )

            main_scores.append(self._jaccard(gold_main, pred_main))
            trans_scores.append(self._jaccard(gold_trans, pred_trans))

        main_avg = sum(main_scores) / len(main_scores)
        trans_avg = sum(trans_scores) / len(trans_scores)
        overall = 0.7 * main_avg + 0.3 * trans_avg

        return main_avg, trans_avg, overall

    @staticmethod
    def age_range_iou(gold_range: List[int], pred_range: List[int]) -> float:
        """IoU for age ranges [start, end]."""
        if not gold_range or not pred_range:
            return 0.0

        g1, g2 = gold_range
        p1, p2 = pred_range
        inter = max(0, min(g2, p2) - max(g1, p1))
        union = max(g2, p2) - min(g1, p1)
        return inter / union if union else 0.0

    def daxian_range_score(self, gold_chart, pred_chart):
        """Average IoU of da-xian age ranges."""
        scores = []
        for palace, ginfo in gold_chart.items():
            gold_range = self._safe_get(ginfo, "大限", "範圍")
            pred_range = self._safe_get(pred_chart.get(palace, {}), "大限", "範圍")
            if gold_range and pred_range:
                scores.append(self.age_range_iou(gold_range, pred_range))
        return sum(scores) / len(scores) if scores else 0.0

    def daxian_ganzhi_accuracy(self, gold_chart, pred_chart):
        """Accuracy of da-xian heavenly stem (gan) and earthly branch (zhi)."""
        matches = 0
        total = 0

        for palace, ginfo in gold_chart.items():
            gdx = self._safe_get(ginfo, "大限", default={}) or {}
            pdx = self._safe_get(pred_chart.get(palace, {}), "大限", default={}) or {}

            if "天干" in gdx:
                total += 1
                if gdx.get("天干") == pdx.get("天干"):
                    matches += 1

            if "地支" in gdx:
                total += 1
                if gdx.get("地支") == pdx.get("地支"):
                    matches += 1

        return matches / total if total else 0.0

    # -----------------------------
    # Star–palace pair extraction
    # -----------------------------
    def _collect_all_stars(self, chart) -> Set[str]:
        stars = set()
        for info in chart.values():
            stars.update(self._safe_get(info, "本命", "主星", default=[]) or [])
            stars.update(self._safe_get(info, "本命", "化曜", default=[]) or [])
        return stars

    def _star_variants(self, star: str) -> Set[str]:
        return {star, self.normalize_text(star)}

    def extract_star_palace_pairs_from_text(self, gold_chart, text, window_size=30):
        """Extract (palace, star) mention pairs from interpretation text."""
        if not text:
            return set()

        norm_text = self.normalize_text(text)
        all_stars = self._collect_all_stars(gold_chart)

        palace_variants = {
            palace: [self.normalize_text(v) for v in self.PALACE_VARIANTS.get(palace, [palace])]
            for palace in gold_chart.keys()
        }
        star_variants = {s: self._star_variants(s) for s in all_stars}

        pairs = set()

        for palace, vars_ in palace_variants.items():
            for v in vars_:
                start = 0
                while True:
                    idx = norm_text.find(v, start)
                    if idx == -1:
                        break

                    window = norm_text[idx: idx + window_size]

                    for star, svars in star_variants.items():
                        if any(sv in window for sv in svars):
                            pairs.add((palace, star))

                    start = idx + 1

        return pairs

    def star_palace_pair_metrics(self, gold_chart, gold_text, pred_text):
        """Compute precision/recall/F1 for extracted star–palace pairs."""
        gold_pairs = self.extract_star_palace_pairs_from_text(gold_chart, gold_text)
        pred_pairs = self.extract_star_palace_pairs_from_text(gold_chart, pred_text)

        if not gold_pairs and not pred_pairs:
            return {"star_palace_pair_precision": 1.0,
                    "star_palace_pair_recall": 1.0,
                    "star_palace_pair_f1": 1.0}

        inter = gold_pairs & pred_pairs

        precision = len(inter) / len(pred_pairs) if pred_pairs else 0.0
        recall = len(inter) / len(gold_pairs) if gold_pairs else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

        return {
            "star_palace_pair_precision": precision,
            "star_palace_pair_recall": recall,
            "star_palace_pair_f1": f1,
        }

    # -----------------------------
    # Topic coverage
    # -----------------------------
    def extract_topic_keywords(self, text):
        norm_text = self.normalize_text(text)
        result = {}

        for topic, concept_dict in self.TOPIC_KEYWORDS.items():
            found = set()
            for concept, forms in concept_dict.items():
                for sf in forms:
                    if self.normalize_text(sf) in norm_text:
                        found.add(concept)
                        break
            if found:
                result[topic] = found

        return result

    def topic_metrics(self, gold_text, pred_text):
        gold_topics = self.extract_topic_keywords(gold_text)
        pred_topics = self.extract_topic_keywords(pred_text)

        if not gold_topics and not pred_topics:
            return {
                "topic_precision": 1.0,
                "topic_recall": 1.0,
                "topic_f1": 1.0,
                "gold_topics": [],
                "pred_topics": [],
                "per_topic_stats": {},
            }

        total_gold = total_pred = total_inter = 0
        per_topic_stats = {}

        for topic in self.TOPIC_KEYWORDS.keys():
            gset = gold_topics.get(topic, set())
            pset = pred_topics.get(topic, set())
            inter = gset & pset

            total_gold += len(gset)
            total_pred += len(pset)
            total_inter += len(inter)

            if not gset and not pset:
                continue

            tp = len(inter) / len(pset) if pset else 0.0
            tr = len(inter) / len(gset) if gset else 0.0
            tf = 2 * tp * tr / (tp + tr) if tp + tr else 0.0

            per_topic_stats[topic] = {
                "gold_keywords": sorted(gset),
                "pred_keywords": sorted(pset),
                "precision": tp,
                "recall": tr,
                "f1": tf,
            }

        precision = total_inter / total_pred if total_pred else 0.0
        recall = total_inter / total_gold if total_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return {
            "topic_precision": precision,
            "topic_recall": recall,
            "topic_f1": f1,
            "gold_topics": list(gold_topics.keys()),
            "pred_topics": list(pred_topics.keys()),
            "per_topic_stats": per_topic_stats,
        }

    # -----------------------------
    # Text similarity
    # -----------------------------
    def cosine_sim(self, gold_text, pred_text):
        emb1 = self.cos_model.encode([gold_text])
        emb2 = self.cos_model.encode([pred_text])
        return float(cosine_similarity(emb1, emb2)[0][0])

    def compute_bertscore(self, gold_text, pred_text):
        _, _, F1 = bertscore([pred_text], [gold_text], lang=self.bert_lang)
        return float(F1[0])

    # -----------------------------
    # Single-sample evaluation
    # -----------------------------
    def evaluate_one(self, birth_data, gold_chart, gold_text, pred_chart, pred_text):
        results = {}

        # Structural metrics
        results["chart_accuracy_exact"] = self.chart_accuracy_exact(gold_chart, pred_chart)
        main_j, trans_j, overall_j = self.chart_star_jaccard(gold_chart, pred_chart)
        results["chart_star_jaccard_main"] = main_j
        results["chart_star_jaccard_trans"] = trans_j
        results["chart_star_jaccard_overall"] = overall_j
        results["daxian_range_iou"] = self.daxian_range_score(gold_chart, pred_chart)
        results["daxian_ganzhi_accuracy"] = self.daxian_ganzhi_accuracy(gold_chart, pred_chart)

        # Star–palace pairs
        results.update(self.star_palace_pair_metrics(gold_chart, gold_text, pred_text))

        # Topics
        results.update(self.topic_metrics(gold_text, pred_text))

        # Text similarity
        results["cosine_similarity"] = self.cosine_sim(gold_text, pred_text)
        results["bertscore"] = self.compute_bertscore(gold_text, pred_text)

        # Weighted grouped scores
        results["overall_structural_score"] = (
            0.25 * results["chart_accuracy_exact"]
            + 0.25 * results["chart_star_jaccard_overall"]
            + 0.25 * results["daxian_range_iou"]
            + 0.25 * results["daxian_ganzhi_accuracy"]
        )

        results["overall_interpretation_content_score"] = (
            0.5 * results["star_palace_pair_f1"]
            + 0.5 * results["topic_f1"]
        )

        results["overall_text_similarity_score"] = (
            0.5 * results["cosine_similarity"]
            + 0.5 * results["bertscore"]
        )

        return results

    # -----------------------------
    # JSONL I/O
    # -----------------------------
    @staticmethod
    def _load_jsonl(path: str):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    @staticmethod
    def _write_jsonl(path: str, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _parse_gold_record(rec):
        birth_data = rec.get("出生資料")
        chart_root = rec.get("命盤", {})
        gold_chart = chart_root.get("命盤", {})
        gold_interpretation = rec.get("解讀", "")
        return birth_data, gold_chart, gold_interpretation

    @staticmethod
    def _parse_pred_record(rec):
        chart_root = rec.get("命盤", {})
        pred_chart = chart_root.get("命盤", {})
        pred_interpretation = rec.get("解讀", "")
        return pred_chart, pred_interpretation

    @staticmethod
    def aggregate_metrics(all_results):
        if not all_results:
            return {}

        agg = {}
        keys = all_results[0].keys()

        for k in keys:
            vals = [
                float(res[k])
                for res in all_results
                if isinstance(res.get(k), (int, float))
            ]
            if vals:
                agg[k] = sum(vals) / len(vals)

        return agg

    # -----------------------------
    # Batch evaluation
    # -----------------------------
    def evaluate_jsonl_files(
        self,
        gold_path: str,
        pred_path: str,
        per_case_output: Optional[str] = None,
        summary_output: Optional[str] = None,
    ):
        gold_records = self._load_jsonl(gold_path)
        pred_records = self._load_jsonl(pred_path)

        if len(gold_records) != len(pred_records):
            raise ValueError(f"gold lines ({len(gold_records)}) != pred lines ({len(pred_records)})")

        all_results = []

        for idx, (g, p) in enumerate(zip(gold_records, pred_records)):
            birth, gold_chart, gold_text = self._parse_gold_record(g)
            pred_chart, pred_text = self._parse_pred_record(p)

            metrics = self.evaluate_one(
                birth_data=birth,
                gold_chart=gold_chart,
                gold_text=gold_text,
                pred_chart=pred_chart,
                pred_text=pred_text,
            )

            metrics_out = {"index": idx}
            metrics_out.update(metrics)
            all_results.append(metrics_out)

        aggregated = self.aggregate_metrics(all_results)

        if per_case_output:
            self._write_jsonl(per_case_output, all_results)

        if summary_output:
            with open(summary_output, "w", encoding="utf-8") as f:
                f.write(json.dumps(aggregated, ensure_ascii=False, indent=2))

        return all_results, aggregated
