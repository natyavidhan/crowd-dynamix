# Crowd Instability Early-Warning System

A real-time simulation and monitoring system for pre-stampede detection using synthetic sensor fusion.

![Architecture](docs/architecture.png)

## Overview

This system simulates crowd behavior and sensor data to demonstrate early detection of crowd instability conditions that could lead to stampede events. It uses:

- **mmWave Radar** (Primary): Velocity variance, stop-go patterns, directional divergence
- **Audio Sensors** (Supporting): Sound energy, distress detection
- **Camera/Vision** (Contextual): Density estimation, optical flow

The **Crowd Instability Index (CII)** provides an explainable, rule-based risk score that rises *before* visible chaos, enabling early intervention.

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

### Installation

```bash
# Clone the repository
cd brainwave

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
npm install
```

### Running

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Open http://localhost:3000 in your browser.

## Demo Scenarios

### 1. Normal Flow
- Start simulation with default settings
- Observe green choke points with low CII

### 2. Density Surge
- Increase "Crowd Inflow" slider to 200%+
- Watch CII rise as density increases

### 3. Opposing Flow
- Enable "Opposing Flow" checkbox
- Creates counter-flow that increases directional divergence
- CII rises due to flow conflicts

### 4. Combined Stress
- Enable both perturbations
- Observe CII approaching red zone
- Note the predictive nature: CII rises before visual chaos

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         BROWSER                                  │
├─────────────────────────────────────────────────────────────────┤
│  React + Leaflet + Zustand                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Map View   │  │  Dashboard  │  │  Controls   │              │
│  │  (Leaflet)  │  │  (Recharts) │  │  (Zustand)  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                          ↑                                       │
│                     WebSocket                                    │
│                          ↓                                       │
├─────────────────────────────────────────────────────────────────┤
│                     Python Backend                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Simulation │  │  Sensors    │  │  CII        │              │
│  │  Engine     │──│  Simulator  │──│  Computer   │              │
│  │  (Agents)   │  │  (mmWave+)  │  │  (Rules)    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  FastAPI + WebSocket (30 Hz state broadcast)                    │
└─────────────────────────────────────────────────────────────────┘
```

## CII Formula

```
CII = w₁·VelocityVariance + w₂·StopGoFrequency + w₃·DirectionalDivergence + w₄·Density
```

Where:
- All factors normalized to [0, 1]
- Default weights: 0.25 each
- Audio acts as a multiplier (up to ×1.2), never triggers alone
- Hysteresis prevents flickering (different thresholds for entering/exiting levels)

## Project Structure

```
brainwave/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt
│   ├── models/
│   │   └── types.py         # Pydantic data models
│   ├── simulation/
│   │   ├── engine.py        # Main simulation loop
│   │   ├── agents.py        # Crowd particle physics
│   │   ├── sensors.py       # Sensor data generation
│   │   └── cii.py           # Risk computation
│   └── utils/
│       └── vector.py        # 2D vector math
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── App.tsx          # Main component
│   │   ├── components/
│   │   │   ├── map/         # Leaflet components
│   │   │   └── dashboard/   # Control & metrics
│   │   ├── store/           # Zustand state
│   │   ├── types/           # TypeScript interfaces
│   │   └── utils/           # Helpers
│   └── index.html
│
└── simulation/
    └── SYSTEM_DESIGN.md     # Detailed technical design
```

## API Endpoints

### REST

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/config` | GET/POST | Simulation config |
| `/control/start` | POST | Start simulation |
| `/control/stop` | POST | Stop simulation |
| `/control/reset` | POST | Reset simulation |
| `/choke-points` | GET/POST/DELETE | Manage choke points |
| `/perturbation/*` | POST | Control perturbations |

### WebSocket

- `WS /ws` - Real-time state at 30 Hz

## License

MIT
