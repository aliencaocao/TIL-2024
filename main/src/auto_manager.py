from random import randint
from typing import Dict, List
from finals_manager import FinalsManager
import requests


class AutoManager(FinalsManager):
    def __init__(self, local_ip: str):
        super().__init__()
        self.local_ip = local_ip

    def run_asr(self, audio_bytes: bytes) -> str:
        print("Running ASR")
        return "asr"

    def run_nlp(self, transcript: str) -> Dict[str, str]:
        print("Running NLP")
        return {
            "target": "airplane",
            "heading": f"{randint(1,360):03}",
            "tool": "surface-to-air missiles",
        }

    def run_vlm(self, image_bytes: bytes, caption: str) -> List[int]:
        print("Running VLM")
        return [0, 0, 0, 0]

    def send_heading(self, heading: str) -> bytes:
        assert heading.isdigit(), "The heading string contains non-digit characters"
        print(f"Sending cannon heading {heading}")
        results = requests.post(
            f"http://{self.local_ip}:5003/send_heading", json={"heading": heading}
        )
        # snapshot of image
        return results.content

    def reset_cannon(self):
        print("Resetting cannon to original position")
        results = requests.post(f"http://{self.local_ip}:5003/reset_cannon")
        print(results.text)
        return results.json()
