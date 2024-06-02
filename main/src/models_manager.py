from base64 import b64encode
from typing import Dict, List
from finals_manager import FinalsManager


class ModelsManager(FinalsManager):
    def __init__(self, local_ip: str):
        super().__init__()
        self.local_ip = local_ip

    async def run_asr(self, audio_bytes: bytes) -> str:
        print("Running ASR")
        audio_str = b64encode(audio_bytes).decode("ascii")
        results = await self.async_post(
            f"http://{self.local_ip}:5001/stt", json={"instances": [{"b64": audio_str}]}
        )
        return results.json()["predictions"][0]

    async def run_nlp(self, transcript: str) -> Dict[str, str]:
        print("Running NLP")
        results = await self.async_post(
            f"http://{self.local_ip}:5002/extract",
            json={"instances": [{"transcript": transcript}]},
        )
        return results.json()["predictions"][0]

    async def send_heading(self, heading: str) -> bytes:
        assert heading.isdigit(), "The heading string contains non-digit characters"
        print(f"Sending cannon heading {heading}")
        results = await self.async_post(
            f"http://{self.local_ip}:5003/send_heading", json={"heading": heading}
        )
        # snapshot of image
        return results.content

    async def reset_cannon(self):
        print("Resetting cannon to original position")
        results = await self.async_post(f"http://{self.local_ip}:5003/reset_cannon")
        return results.json()

    async def run_vlm(self, image_bytes: bytes, caption: str) -> List[int]:
        print("Running VLM")
        image_str = b64encode(image_bytes).decode("ascii")
        results = await self.async_post(
            f"http://{self.local_ip}:5004/identify",
            json={"instances": [{"b64": image_str, "caption": caption}]},
        )
        return results.json()["predictions"][0]
