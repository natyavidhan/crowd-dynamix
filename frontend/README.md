# Crowd Simulation Frontend

React + TypeScript + Leaflet frontend for the crowd instability monitoring system.

## Setup

```bash
cd frontend
npm install
```

## Development

```bash
# Make sure backend is running first
cd ../backend
uvicorn main:app --reload --port 8000

# Then start frontend
cd ../frontend
npm run dev
```

The frontend will be available at http://localhost:3000

## Build

```bash
npm run build
```

Output will be in the `dist` folder.

## Architecture

```
src/
├── App.tsx              # Main application
├── main.tsx             # Entry point
├── index.css            # Global styles
├── vite-env.d.ts        # Vite types
├── components/
│   ├── map/
│   │   ├── MapView.tsx      # Leaflet map container
│   │   ├── ChokePointLayer  # Risk zone visualization
│   │   └── AgentCanvas      # Crowd particle rendering
│   └── dashboard/
│       ├── ControlPanel     # Sim controls
│       ├── ChokePointList   # CP overview
│       └── ChokePointCard   # CP details & charts
├── store/
│   └── simulation.ts    # Zustand state store
├── types/
│   └── simulation.ts    # TypeScript interfaces
└── utils/
    ├── colors.ts        # Color utilities
    └── coordinates.ts   # Geo/sim coord conversion
```

## Features

- Real-time map visualization with OSM tiles
- Crowd particle rendering on canvas (500+ agents @ 60fps)
- Choke point risk overlay (green → yellow → red)
- Live CII charts with time-series history
- Explainable risk factor breakdown
- Simulation controls (start/stop, speed, inflow)
- Perturbation triggers (opposing flow, density surge)
