import json
from typing import Dict, List
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
from scoring.nlp_eval import nlp_eval
from dotenv import load_dotenv
import os

load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")


def main():
    # input_dir = Path(f"/home/jupyter/{TEAM_TRACK}")
    input_dir = Path(f"../../data/{TEAM_TRACK}/train")
    # results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(input_dir / "nlp.jsonl", "r") as f:
        instances = [json.loads(line.strip()) for line in f if line.strip() != ""]

    results = run_batched(instances)
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "nlp_results.csv", index=False)
    # calculate eval
    eval_result = nlp_eval(
        [result["truth"] for result in results],
        [result["prediction"] for result in results],
    )
    print(f"NLP result: {eval_result}")


def run_batched(
    instances: List[Dict[str, str | int]], batch_size: int = 64
) -> List[Dict[str, str | int]]:
    # split into batches
    results = []
    for index in tqdm(range(0, len(instances), batch_size)):
        _instances = instances[index : index + batch_size]
        response = requests.post(
            "http://localhost:5002/extract",
            data=json.dumps(
                {
                    "instances": [
                        {"key": _instance["key"], "transcript": _instance["transcript"]}
                        for _instance in _instances
                    ]
                }
            ),
        )
        _results = response.json()["predictions"]
        results.extend(
            [
                {
                    "key": _instances[i]["key"],
                    "truth": {
                        field: _instances[i][field]
                        for field in ("heading", "target", "tool")
                    },
                    "prediction": _results[i],
                }
                for i in range(len(_instances))
            ]
        )
    return results


if __name__ == "__main__":
    main()
