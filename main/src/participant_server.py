import asyncio
import json
import os
from urllib.parse import quote

import websockets

from finals_manager import FinalsManager
from auto_manager import AutoManager
from mock_manager import MockManager
from models_manager import ModelsManager

TEAM_NAME = os.getenv("TEAM_NAME", "Team Name")
LOCAL_IP = os.getenv("LOCAL_IP", "0.0.0.0")
SERVER_IP = os.getenv("COMPETITION_SERVER_IP", "host.docker.internal")
SERVER_PORT = os.getenv("COMPETITION_SERVER_PORT", "8000")

manager: FinalsManager = ModelsManager(LOCAL_IP)
# manager: FinalsManager = AutoManager(LOCAL_IP)
# manager: FinalsManager = MockManager()


async def server():
    index = 0
    async for websocket in websockets.connect(
        quote(f"ws://{SERVER_IP}:{SERVER_PORT}/ws/{TEAM_NAME}", safe="/:"),
        max_size=2**24,
    ):
        print(f"connecting to competition server {SERVER_IP} at port {SERVER_PORT}")
        try:
            while True:
                # should be receiving either audio bytes for asr, or "done!" message
                socket_input = await websocket.recv()
                if type(socket_input) is str:
                    # handle either done or healthcheck
                    data = json.loads(socket_input)
                    if data["status"] == "done":
                        print("done!")
                        break
                    else:
                        await manager.send_result({"health": "ok"})
                        continue
                print(f"run {index}")
                # ASR
                transcript = manager.run_asr(socket_input)
                print(transcript)
                # NLP
                qa_ans = manager.run_nlp(transcript)
                print(qa_ans)
                query = qa_ans["target"]
                # autonomy
                try:
                    image = manager.send_heading(qa_ans["heading"])
                except AssertionError as e:
                    # if heading is wrong, get image of scene at default heading 000
                    print(e)
                    image = manager.send_heading("000")
                # VLM
                vlm_results = manager.run_vlm(image, query)
                print(vlm_results)
                # submit results and reset
                await manager.send_result(
                    websocket,
                    {"asr": transcript, "nlp": qa_ans, "vlm": vlm_results},
                )
                manager.reset_cannon()
                print(f"done run {index}")
                index += 1
        except websockets.ConnectionClosed:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
        else:
            break


if __name__ == "__main__":
    asyncio.run(server())
