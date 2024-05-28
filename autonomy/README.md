# TIL-Autonomy

This is the container responsible for controlling your robot, and should at minimum be able to handle:

- turning to a particular heading, and
- taking a snapshot from the simulator.

The main entrypoint is `autonomy.py`, which uses all the OS environment variables passed in from the `.env` through the `docker-compose.yml` in the parent directory to configure the required connections (e.g. to the robot, to the competition server, etc). The other files are:

- `environment.py` defines an abstract class of functions that the other two environments must fulfill. It handles the WebSocket connection to the competition server autonomy endpoint.
- `robot_env.py` defines the autonomy environment controlling the actual robot. The robot's gimbal rotation is used to update the simulator
- `sim_env.py` defines the autonomy environment controlling the simulator directly.

## Setup

You can build the autonomy Docker container individually with `docker build -t autonomy .` or you can set up the whole system by running `docker compose up` from the main directory.

## Notes

The pitch is set to 15 because an air defense turret should aim upwards. Any pitch rotations you make will not modify the display on the simulator, and are thus unnecessary.
