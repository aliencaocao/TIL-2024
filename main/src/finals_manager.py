import json
import websockets
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class FinalsManager(ABC):
    def __init__(self):
        print("initializing participant finals server manager")

    def send_result(
        self, websocket: websockets.WebSocketClientProtocol, data: Dict[str, Any]
    ):
        return websocket.send(json.dumps(data))

    @abstractmethod
    def run_asr(self, audio_bytes: bytes) -> str:
        raise NotImplemented

    @abstractmethod
    def run_nlp(self, transcript: str) -> Dict[str, str]:
        raise NotImplemented

    @abstractmethod
    def send_heading(self, heading: str) -> bytes:
        raise NotImplemented

    @abstractmethod
    def reset_cannon(self) -> None:
        raise NotImplemented

    def run_vlm(self, image_bytes: bytes, caption: str) -> List[int]:
        raise NotImplemented
