from asyncio import sleep
from environment import Environment
import json


class SimEnv(Environment):
    """
    This class abstracts out the robomaster code for testing.
    """

    def __init__(self, uri: str) -> None:
        super().__init__(uri)

        self.update_freq = 20
        self.velocity = 60 / self.update_freq
        # initialize yaw
        self.camera_yaw = 0

    async def update_sim(self) -> None:
        print("sending data")
        await self.send_websocket(
            json.dumps({"type": "update", "yaw": self.camera_yaw})
        )

    ## MOVEMENT CODE
    async def pan_cannon(self, change):
        """Pans the cannon in the horizontal direction"""
        changed = 0
        while True:
            await sleep(1 / self.update_freq)
            if changed < change:
                if changed + self.velocity > change:
                    diff = change - changed
                else:
                    diff = self.velocity
            else:
                if changed + self.velocity < change:
                    diff = change - changed
                else:
                    diff = -self.velocity

            # Update yaw here
            self.camera_yaw += diff
            changed += diff
            if self.camera_yaw > 0:
                self.camera_yaw = min(self.camera_yaw, self.camera_yaw_max)
            else:
                self.camera_yaw = max(self.camera_yaw, self.camera_yaw_min)
            await self.update_sim()
            if changed == change:
                break

    async def reset_pan_cannon(self):
        await self.pan_cannon(-self.camera_yaw)

    def stop_cannon(self):
        """Stop the cannon and reset the velocity"""
        pass

    async def exit(self):
        """Clean up and exit"""
        await self._close_websocket()

    def get_yaw(self):
        return self.camera_yaw
