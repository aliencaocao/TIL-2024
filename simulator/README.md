# BrainHack TIL-AI 24 Simulator Environment

## Building the simulator

You can build the simulator + competition server from the top-level directory by running:

```bash
docker build -t competition .
```

## Running the simulator

You can run the simulator + competition server from the top-level directory by running:

```bash
docker run -p 8000:8000 competition
```

Then, you should see the simulator when you access `localhost:8000` from your browser machine.
