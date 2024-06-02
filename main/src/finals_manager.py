import json
import websockets
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import httpx


class FinalsManager(ABC):
    def __init__(self):
        print("initializing participant finals server manager")
        self.client = httpx.AsyncClient()

    async def exit(self):
        await self.client.aclose()

    async def async_post(self, endpoint: str, json: Optional[Dict] = None):
        return await self.client.post(endpoint, json=json)

    async def send_result(
        self, websocket: websockets.WebSocketClientProtocol, data: Dict[str, Any]
    ):
        return await websocket.send(json.dumps(data))

    @abstractmethod
    async def run_asr(self, audio_bytes: bytes) -> str:
        raise NotImplemented

    @abstractmethod
    async def run_nlp(self, transcript: str) -> Dict[str, str]:
        raise NotImplemented

    @abstractmethod
    async def send_heading(self, heading: str) -> bytes:
        raise NotImplemented

    @abstractmethod
    async def reset_cannon(self) -> None:
        raise NotImplemented

    @abstractmethod
    async def run_vlm(self, image_bytes: bytes, caption: str) -> List[int]:
        raise NotImplemented
