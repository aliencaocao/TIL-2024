from __future__ import annotations

import os, time
from contextlib import asynccontextmanager

from environment import Environment
from fastapi import FastAPI, HTTPException, Request, Response
from robot_env import RobotEnv
from sim_env import SimEnv

import PIL.Image as Image
import io

env: Environment | None = None
TEAM_NAME = os.getenv("TEAM_NAME", "Team Name")
SERVER_IP = os.getenv("COMPETITION_SERVER_IP", "host.docker.internal")
SERVER_PORT = os.getenv("COMPETITION_SERVER_PORT", "8000")
ROBOT_IP = os.getenv("ROBOT_IP", "192.168.10.10")
ROBOT_SN = os.getenv("ROBOT_SN", "ABC123")
USE_ROBOT = os.getenv("USE_ROBOT", "false").lower() in ["true", "1", "t", "y", "yes"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    # initialize websocket with competition server
    if USE_ROBOT:
        print("using robot")
        assert ROBOT_IP, "No robot IP provided"
        assert ROBOT_SN, "No robot serial number provided"
        print(
            f"robot ip: {ROBOT_IP}; robot sn: {ROBOT_SN}; team name: {TEAM_NAME}; server: {SERVER_IP}:{SERVER_PORT}"
        )
        env = RobotEnv(
            uri=f"ws://{SERVER_IP}:{SERVER_PORT}/ws_auto/{TEAM_NAME}",
            robot_sn=ROBOT_SN,
            robot_ip=ROBOT_IP,
            local_ip="0.0.0.0",
        )
    else:
        print(f"using sim env for server {SERVER_IP}:{SERVER_PORT}")
        env = SimEnv(f"ws://{SERVER_IP}:{SERVER_PORT}/ws_auto/{TEAM_NAME}")
    await env._init_websocket()
    yield
    await env.exit()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "This is the API service for TIL-autonomy"}


@app.get("/health")
async def health():
    if env.health():
        return {"health": "ok"}
    else:
        return HTTPException(503, "websocket not initialized")

count = 0
@app.post("/send_heading")
async def send_heading(request: Request):
    global count
    request_dict = await request.json()

    heading = request_dict["heading"]
    print(request_dict)
    # TODO: fill in here
    # depends on how your team would like to implement the robotics component
    heading = int(heading)
    if heading > 180:
        heading -= 360
    # rotate to heading
    await env.pan_cannon(heading)
    print("taking snapshot")
    b_image: bytes = await env.take_snapshot()
    
    savepath = str(count) + ".jpg"
    image = Image.open(io.BytesIO(b_image))
    image.save(savepath)
    count += 1
    
    return Response(content=b_image, media_type="image/jpeg")


# optional, depends on how your team would like to implement the robotics component
@app.post("/reset_cannon")
async def reset_cannon():
    # TODO: fill in here
    await env.reset_pan_cannon()
    return {"message": "done"}
