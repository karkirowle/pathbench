from typing import Dict, Optional

from pathbench.evaluator import LookupEvaluator


class Spk2AgeEvaluator(LookupEvaluator):
    """An evaluator that uses a pre-computed spk2age mapping."""

    def __init__(self, spk2age: Dict[str, float], utt2spk: Dict[str, str]):
        self.spk2age = spk2age
        self.utt2spk = utt2spk

    def score(self, utterance_id: str) -> Optional[float]:
        speaker_id = self.utt2spk.get(utterance_id)
        if speaker_id:
            return self.spk2age.get(speaker_id)
        return None
