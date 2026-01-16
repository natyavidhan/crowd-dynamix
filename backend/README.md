# Crowd Simulation Backend

Real-time crowd instability simulation with sensor fusion for early warning systems.

## Requirements

- Python 3.11+
- FastAPI
- NumPy
- Pydantic

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Running

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

## API Endpoints

### REST

- `GET /` - Health check
- `GET /config` - Get simulation config
- `POST /config` - Update simulation config
- `POST /control/start` - Start simulation
- `POST /control/stop` - Stop simulation
- `POST /control/reset` - Reset simulation
- `POST /map/origin` - Set map origin
- `GET /choke-points` - List choke points
- `POST /choke-points` - Add choke point
- `DELETE /choke-points/{id}` - Remove choke point
- `POST /perturbation/opposing-flow` - Toggle opposing flow
- `POST /perturbation/density-surge` - Toggle density surge
- `POST /perturbation/inflow` - Set inflow multiplier
- `POST /perturbation/speed` - Set speed multiplier

### WebSocket

- `WS /ws` - Real-time state updates at 30 Hz

## Architecture

```
backend/
├── main.py              # FastAPI app
├── models/
│   └── types.py         # Pydantic models
├── simulation/
│   ├── engine.py        # Main simulation loop
│   ├── agents.py        # Crowd particle physics
│   ├── sensors.py       # Sensor data simulation
│   └── cii.py           # Risk computation
└── utils/
    └── vector.py        # 2D vector math
```
