from asyncio import sleep
from typing import Dict, List
from finals_manager import FinalsManager


class MockManager(FinalsManager):
    def __init__(self):
        super().__init__()

    async def run_asr(self, audio_bytes: bytes) -> str:
        print("Running ASR")
        return "asr"

    async def run_nlp(self, transcript: str) -> Dict[str, str]:
        print("Running NLP")
        return {
            "target": "airplane",
            "heading": "180",
            "tool": "surface-to-air missiles",
        }

    async def run_vlm(self, image_bytes: bytes, caption: str) -> List[int]:
        print("Running VLM")
        return [0, 0, 0, 0]

    async def send_heading(self, heading: str) -> bytes:
        print(f"Sending cannon heading {heading}")
        await sleep(1)
        return bytes()

    async def reset_cannon(self) -> None:
        print("Resetting cannon to original position")
        return {}
