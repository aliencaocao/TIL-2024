import jiwer
from typing import List

wer_transforms = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def asr_eval(truth: List[str], hypothesis: List[str]) -> float:
    result = jiwer.wer(
        truth,
        hypothesis,
        truth_transform=wer_transforms,
        hypothesis_transform=wer_transforms,
    )
    return 1 - result
