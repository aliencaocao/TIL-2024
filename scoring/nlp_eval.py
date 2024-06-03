from statistics import mean
from typing import Dict, List, Tuple

from rouge_score import rouge_scorer

ROUGEL = "rougeL"
scorer = rouge_scorer.RougeScorer([ROUGEL], use_stemmer=True)

keys = ("heading", "target", "tool")


def score(key: str, ref: str, hyp: str) -> float:
    # exact match for heading
    if key == "heading":
        if ref != hyp:
            print(ref, hyp)
        return 1.0 if ref == hyp else 0.0
    # ROUGE-L for everything else
    else:
        score = scorer.score(ref, hyp)[ROUGEL]
        if score.fmeasure != 1:
            print(key, ref, hyp, score.fmeasure)
        return score.fmeasure


def nlp_eval(truth: List[Dict[str, str]], hypothesis: List[Dict[str, str]]) -> float:
    results = []
    for ref, hyp in zip(truth, hypothesis):
        results.append(mean(score(key, ref[key], hyp[key]) for key in keys))
    return mean(results)


def nlp_eval_detailed(truth: List[Dict[str, str]], hypothesis: List[Dict[str, str]]) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    scores = {key: [] for key in keys}
    results = []
    for ref, hyp in zip(truth, hypothesis):
        results.append(mean(score(key, ref[key], hyp[key]) for key in keys))
        for key in keys:
            scores[key].append(score(key, ref[key], hyp[key]))
    scores_by_truth_key = {t['key']: {key: scores[key][i] for key in keys} for i, t in enumerate(truth)}
    return mean(results), {key: mean(value) for key, value in scores.items()}, scores_by_truth_key


if __name__ == "__main__":
    import orjson
    import json
    result_filename = 'gorilla-openfunctions-v2-TIL24-r16-a16-ctx768-v2-5bit-hb6-3500samples'
    if '3500samples' in result_filename:  # train set full
        with open("../data/nlp.jsonl", "r") as f:
            truth = [orjson.loads(line.strip()) for line in f if line.strip() != ""]
    else:
        with open("../nlp/src/gpt_eval.jsonl", "r") as f:
            truth = [orjson.loads(line.strip()) for line in f if line.strip() != ""]
    with open(f'../nlp/src/eval_outputs/{result_filename}.json', 'r') as f:
        hyp = [orjson.loads(line.strip()) for line in f if line.strip() != ""]
    hyp = hyp[0]
    assert len(truth) == len(hyp)
    mean_score, eval_result, scores_by_truth_key = nlp_eval_detailed(truth, hyp)
    print(f"NLP mean score: {mean_score}")
    print(f"NLP detailed score: {eval_result}")
    with open(f'../nlp/src/eval_results/{result_filename}_score.json', 'w+') as f:
        f.write(json.dumps(scores_by_truth_key))
