from abc import ABC, abstractmethod
from typing import Any, Tuple
from urllib.parse import quote

import websockets


class Environment(ABC):
    ## initialize websocket
    @abstractmethod
    def __init__(self, uri: str) -> None:
        self.uri = quote(uri, safe="/:")
        self._init_params()

    def _init_params(self) -> None:
        # define camera movement limits
        self.camera_yaw_max = 180  # in degrees (actual max controllable yaw is +250)
        self.camera_yaw_min = -180  # in degrees (actual min controllable yaw is -250)

    async def _init_websocket(self) -> None:
        print(self.uri)
        self.websocket = await websockets.connect(self.uri, max_size=2**24)

    def health(self) -> bool:
        return self.websocket is not None

    async def _close_websocket(self) -> None:
        await self.websocket.close()

    async def send_websocket(self, data: str) -> None:
        await self.websocket.send(data)

    async def take_snapshot(self) -> Any:
        await self.websocket.send('{"type": "snapshot"}')
        return await self.websocket.recv()

    @abstractmethod
    async def pan_cannon(self, change: float) -> None:
        raise NotImplemented

    @abstractmethod
    async def reset_pan_cannon(self) -> None:
        raise NotImplemented

    @abstractmethod
    def stop_cannon(self) -> None:
        raise NotImplemented

    @abstractmethod
    def exit(self) -> None:
        raise NotImplemented

    @abstractmethod
    def get_yaw(self) -> float:
        raise NotImplemented

    def get_yaw_limits(self) -> Tuple[float, float]:
        return (self.camera_yaw_min, self.camera_yaw_max)
