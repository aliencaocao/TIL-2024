# TIL-24 Finals Main Orchestration Server

## Overview

The main entrypoint is `participant_server.py`, which when started, instantiates a WebSocket connection with the competition server and begins listening for input ASR data. It then triggers the other models by calling them through HTTP endpoints. If you wish to change the way your models work and interface, you are welcome to; just know that any changes you make come at your own peril.

Note that there are 3 different "managers" you can use to test your finals code, but your final version should most likely be built using `models_manager.py`; the others are meant to help you test your solution in the absence of the `robomaster` turret hardware or models running locally.

- `mock_manager.py` uses mock versions of everything
- `auto_manager.py` enables autonomy and not the models, so you can test your robotics solution in isolation.
- `models_manager.py` uses the real models and the real autonomy code, so you can test everything altogether.

## Reminders

Make sure to configure `.env` in parent directory accordingly.

You should be able to just `docker compose up` from the `boilerplate` directory, if you need to manually force a rebuild there's `docker compose up --build`.

You will need a competition server to be running first, or the services will fail to start. The `docker-compose.yml` should be configured to automatically do so.
