from statistics import mean
from rouge_score import rouge_scorer
from typing import List, Dict

ROUGEL = "rougeL"
scorer = rouge_scorer.RougeScorer([ROUGEL], use_stemmer=True)

keys = ("heading", "target", "tool")


def score(key: str, ref: str, hyp: str) -> float:
    # exact match for heading
    if key == "heading":
        return 1.0 if ref == hyp else 0.0
    # ROUGE-L for everything else
    else:
        score = scorer.score(ref, hyp)[ROUGEL]
        return score.fmeasure


def nlp_eval(truth: List[Dict[str, str]], hypothesis: List[Dict[str, str]]) -> float:
    results = []
    for ref, hyp in zip(truth, hypothesis):
        results.append(mean(score(key, ref[key], hyp[key]) for key in keys))
    return mean(results)
