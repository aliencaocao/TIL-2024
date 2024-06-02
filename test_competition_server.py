import asyncio
import json
import logging
import os
from time import time
from typing import Any, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from scoring.nlp_eval import nlp_eval
from scoring.vlm_eval import vlm_eval
from starlette.responses import FileResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

prefix: str = "results/"
TEAM_NAME: str = os.getenv("TEAM_NAME", "team-name")

# filepath to load testcase data from
filepath = "simulator/data/data.json"
with open(filepath) as f:
    testcases = json.load(f)
os.makedirs(prefix, exist_ok=True)

app = FastAPI()
app.mount("/simulator", StaticFiles(directory="simulator"), name="simulator")


# serve simulator html
@app.get("/")
async def read_index():
    return FileResponse("simulator/index.html")


@app.get("/health")
def health():
    return {"health": "ok"}


# websocket connection manager
class MockConnectionManager:
    def __init__(self):
        self.simulator_connection: WebSocket | None = None
        self.team_connection: WebSocket | None = None
        self.team_bbox: List[int] = [0, 0, 0, 0]
        self.autonomy_connection: WebSocket | None = None

    async def simulator_connect(self, websocket: WebSocket):
        if self.simulator_connection == None:
            await websocket.accept()
            self.simulator_connection = websocket
            await websocket.send_json(
                {
                    "type": "teams",
                    "teams": [TEAM_NAME],
                }
            )
        else:
            logger.info("there is already a simulator ws connection")
            try:
                await self.simulator_connection.close()
            except Exception as e:
                logger.exception(e)
            await websocket.accept()
            self.simulator_connection = websocket

    def simulator_disconnect(self):
        logger.info("disconnecting previous connection")
        self.simulator_connection = None

    async def team_connect(self, websocket: WebSocket):
        if self.team_connection == None:
            await websocket.accept()
            self.team_connection = websocket
        else:
            # await websocket.close(
            #     reason=f"There is already a team connected with id {team_id}!"
            # )
            await self.team_connection.close()
            await websocket.accept()
            self.team_connection = websocket

    async def autonomy_connect(self, websocket: WebSocket):
        if self.autonomy_connection == None:
            await websocket.accept()
            self.autonomy_connection = websocket
        else:
            # await websocket.close(
            #     reason=f"There is already a team connected with id {team_id}!"
            # )
            await self.autonomy_connection.close()
            await websocket.accept()
            self.autonomy_connection = websocket

    async def update_simulator(self, websocket: WebSocket, team_name: int, data: dict):
        if self.simulator_connection is not None:
            await self.simulator_connection.send_json({"name": team_name, **data})
            if data["type"] == "snapshot":
                bbox = await self.simulator_connection.receive_json()
                snapshot = await self.simulator_connection.receive_bytes()
                await websocket.send_bytes(snapshot)
                if bbox["valid"]:
                    self.team_bbox = [
                        bbox[side] for side in ("left", "top", "width", "height")
                    ]
                else:
                    self.team_bbox = None
                logger.info("sending snapshot and writing file")
                with open(f"{prefix}team_{team_name}_snapshot.jpg", "wb") as file:
                    file.write(snapshot)

    def disconnect(self, team_name: str):
        self.team_connection = None
        self.autonomy_connection = None


manager = MockConnectionManager()


# websocket connection for simulator
@app.websocket("/ws_sim")
async def simulator_websocket(websocket: WebSocket):
    await manager.simulator_connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("disconnected")
    finally:
        manager.simulator_disconnect()


@app.websocket("/ws/{team_name}")
async def team_endpoint(websocket: WebSocket, team_name: str):
    await manager.team_connect(websocket)
    try:
        responses: List[Dict[str, Any]] = []
        for case in testcases:
            with open(case["audio"], "rb") as file:
                audio_bytes = file.read()
            # determine time between start and end
            start_time = time()
            await websocket.send_bytes(audio_bytes)
            results = await websocket.receive_json()
            # evaluate responses
            elapsed = time() - start_time
            logger.info(f"Team {team_name} took {elapsed:.3f}s")
            results["bbox"] = manager.team_bbox
            results["truth"] = case["truth"]
            results["elapsed"] = elapsed
            results["nlp_score"] = nlp_eval([case["truth"]], [results["nlp"]])
            results["vlm_score"] = vlm_eval([results["bbox"]], [results["vlm"]])
            results["perf_score"] = 1 - min(30, elapsed) / 30
            results["score"] = (
                0.45 * results["nlp_score"]
                + 0.45 * results["vlm_score"]
                + 0.1 * results["perf_score"]
            )
            logger.info(results)
            responses.append(results)
            await manager.simulator_connection.send_json({"type": "switch"})
        await websocket.send_json({"status": "done"})
        # write responses
        with open(prefix + f"team_{team_name}_results.jsonl", "w") as f:
            for response in responses:
                f.write(json.dumps(response) + "\n")
    except WebSocketDisconnect:
        logger.info(f"Team {team_name} disconnected")
    finally:
        manager.disconnect(team_name)


@app.websocket("/ws_auto/{team_name}")
async def autonomy_endpoint(websocket: WebSocket, team_name: str):
    await manager.autonomy_connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await manager.update_simulator(websocket, team_name, data)
    except WebSocketDisconnect:
        manager.disconnect(team_name)
