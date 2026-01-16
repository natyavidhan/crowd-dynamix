# Crowd Instability Early-Warning System: Technical Design Document

**Version**: 1.0  
**Date**: January 2026  
**Purpose**: Hackathon demo for pre-stampede detection using simulated sensor fusion

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BROWSER (CLIENT)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │   MAP RENDERER   │    │  SIMULATION      │    │   DASHBOARD      │       │
│  │   (Leaflet)      │◄───│  ENGINE          │───►│   PANEL          │       │
│  │                  │    │                  │    │                  │       │
│  │  - OSM tiles     │    │  - Agent system  │    │  - CII gauges    │       │
│  │  - Choke points  │    │  - Sensor sim    │    │  - Time series   │       │
│  │  - Heat overlay  │    │  - CII compute   │    │  - Controls      │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│           ▲                       │                       ▲                  │
│           │                       ▼                       │                  │
│           │              ┌──────────────────┐             │                  │
│           └──────────────│   STATE STORE    │─────────────┘                  │
│                          │   (Zustand)      │                                │
│                          │                  │                                │
│                          │  - Agents[]      │                                │
│                          │  - ChokePoints[] │                                │
│                          │  - SensorData[]  │                                │
│                          │  - SimConfig     │                                │
│                          └──────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Breakdown

| Component | Technology | Responsibility |
|-----------|------------|----------------|
| Map Renderer | Leaflet + React-Leaflet | OSM tiles, choke point rendering, heat overlays |
| Simulation Engine | Pure TypeScript | Agent movement, sensor data generation, CII computation |
| Dashboard Panel | React + Recharts | Real-time metrics display, controls |
| State Store | Zustand | Centralized state with subscriptions |

### 1.3 Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  User Input │────►│  Sim Engine │────►│  State      │────►│  Renderers  │
│  (controls) │     │  (60 FPS)   │     │  Update     │     │  (React)    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Sensor     │
                    │  Samplers   │
                    │  (10 Hz)    │
                    └─────────────┘
```

**Key timing considerations:**
- Simulation tick: 60 FPS (16.67ms) for smooth animation
- Sensor sampling: 10 Hz (100ms) for realistic data streams
- CII computation: 1 Hz (1000ms) to allow pattern detection over time windows
- React re-render: Decoupled via Zustand subscriptions (only when state changes)

---

## 2. Data Models

### 2.1 Geographic Types

```typescript
interface GeoPoint {
  lat: number;
  lng: number;
}

interface GeoBounds {
  north: number;
  south: number;
  east: number;
  west: number;
}

// Pixel coordinates for simulation (map projected)
interface SimPoint {
  x: number;  // meters from origin
  y: number;  // meters from origin
}
```

### 2.2 Choke Point Model

```typescript
type SensorType = 'mmwave' | 'audio' | 'camera';

interface ChokePoint {
  id: string;
  name: string;
  center: GeoPoint;
  
  // Geometry - support both circular and polygonal regions
  geometry: 
    | { type: 'circle'; radius: number }  // radius in meters
    | { type: 'polygon'; vertices: GeoPoint[] };
  
  // Sensor configuration
  sensors: {
    type: SensorType;
    enabled: boolean;
    confidence: number;  // 0-1, simulates sensor reliability
  }[];
  
  // Computed risk state (updated by simulation)
  riskState: {
    cii: number;           // Crowd Instability Index [0, 1]
    trend: 'stable' | 'rising' | 'falling';
    lastUpdated: number;   // timestamp
  };
}
```

### 2.3 Agent (Crowd Particle) Model

```typescript
type AgentState = 'moving' | 'slowing' | 'stopped' | 'pushing';

interface Agent {
  id: number;
  position: SimPoint;
  velocity: SimPoint;        // current velocity vector
  targetVelocity: SimPoint;  // desired velocity (for steering)
  
  // Path following
  waypoints: SimPoint[];
  currentWaypointIndex: number;
  
  // Behavioral state
  state: AgentState;
  patience: number;          // 0-1, decreases when blocked
  
  // For visualization
  radius: number;            // collision radius (0.3m typical)
}
```

### 2.4 Sensor Data Models

```typescript
// mmWave radar output (primary sensor)
interface MmWaveSensorData {
  timestamp: number;
  chokePointId: string;
  
  // Velocity metrics
  avgVelocity: number;           // m/s, magnitude
  velocityVariance: number;      // statistical variance
  dominantDirection: number;     // radians, 0 = north
  directionalDivergence: number; // 0-1, 0=all same dir, 1=chaos
  
  // Stop-go detection
  stopGoFrequency: number;       // events per minute
  stationaryRatio: number;       // fraction of stationary targets
}

// Audio sensor output (supporting)
interface AudioSensorData {
  timestamp: number;
  chokePointId: string;
  
  soundEnergyLevel: number;      // dB normalized to 0-1
  spikeDetected: boolean;
  spikeIntensity: number;        // 0-1 if spike detected
  
  // Coarse classification (simulated)
  audioCharacter: 'ambient' | 'loud' | 'distressed';
}

// Camera sensor output (contextual)
interface CameraSensorData {
  timestamp: number;
  chokePointId: string;
  
  crowdDensity: number;          // people per m²
  occlusionPercentage: number;   // 0-1, how much is blocked
  opticalFlowConsistency: number; // 0-1, 1=uniform flow
}

// Aggregated sensor reading for a choke point
interface AggregatedSensorData {
  chokePointId: string;
  timestamp: number;
  
  mmwave: MmWaveSensorData | null;
  audio: AudioSensorData | null;
  camera: CameraSensorData | null;
  
  // Computed
  overallConfidence: number;     // weighted sensor confidence
}
```

### 2.5 Simulation Configuration

```typescript
interface SimulationConfig {
  // Time control
  running: boolean;
  speedMultiplier: number;       // 1x, 2x, 5x
  
  // Scenario parameters
  baseInflowRate: number;        // agents per second
  currentInflowMultiplier: number;
  
  // Active perturbations
  perturbations: {
    opposingFlow: boolean;
    blockedExits: string[];      // exit IDs that are blocked
    densitySurge: boolean;
  };
  
  // Map bounds (for coordinate conversion)
  mapBounds: GeoBounds;
  metersPerPixel: number;
}
```

### 2.6 Time Series Buffer

```typescript
interface TimeSeriesBuffer<T> {
  maxLength: number;             // e.g., 300 samples = 30 seconds at 10Hz
  samples: {
    timestamp: number;
    value: T;
  }[];
  
  // Methods
  push(value: T, timestamp: number): void;
  getWindow(durationMs: number): T[];
  getLatest(): T | null;
}
```

---

## 3. Core Simulation Logic

### 3.1 Agent Movement (Steering Behaviors)

The crowd simulation uses a layered steering approach:

```
Agent Velocity = Σ weighted forces:
  + Path Following Force     (highest priority)
  + Collision Avoidance      (high priority)
  + Separation Force         (medium priority)  
  + Alignment Force          (low priority, mimics crowd flow)
```

**Pseudocode: Agent Update**

```typescript
function updateAgent(agent: Agent, neighbors: Agent[], dt: number): void {
  // 1. Compute target velocity from path
  const pathForce = computePathFollowingForce(agent);
  
  // 2. Collision avoidance (predictive)
  const avoidanceForce = computeCollisionAvoidance(agent, neighbors);
  
  // 3. Separation (personal space)
  const separationForce = computeSeparationForce(agent, neighbors);
  
  // 4. Alignment (flow with crowd)
  const alignmentForce = computeAlignmentForce(agent, neighbors);
  
  // 5. Combine with weights
  const totalForce = Vector.add(
    Vector.scale(pathForce, 1.0),
    Vector.scale(avoidanceForce, 2.0),      // high weight
    Vector.scale(separationForce, 1.5),
    Vector.scale(alignmentForce, 0.5)
  );
  
  // 6. Apply acceleration limits
  const acceleration = Vector.clamp(totalForce, MAX_ACCELERATION);
  
  // 7. Update velocity with damping
  agent.velocity = Vector.add(
    Vector.scale(agent.velocity, 0.95),     // friction
    Vector.scale(acceleration, dt)
  );
  
  // 8. Clamp to max speed
  agent.velocity = Vector.clampMagnitude(agent.velocity, MAX_SPEED);
  
  // 9. Update position
  agent.position = Vector.add(
    agent.position,
    Vector.scale(agent.velocity, dt)
  );
  
  // 10. Update behavioral state
  agent.state = classifyAgentState(agent);
}
```

**Key parameters:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| MAX_SPEED | 1.4 m/s | Average walking speed |
| MAX_ACCELERATION | 2.0 m/s² | Comfortable acceleration |
| PERSONAL_SPACE | 0.5 m | Minimum comfortable distance |
| NEIGHBOR_RADIUS | 3.0 m | Perception range |
| COLLISION_HORIZON | 2.0 s | Look-ahead for collision prediction |

### 3.2 Crowd Behavior Patterns

**Pattern: Stop-Go Wave**

Emerges naturally when:
- Downstream agents slow/stop
- Upstream agents must decelerate
- Wave propagates backward through crowd

Detection:
```typescript
function detectStopGoWave(velocityHistory: number[]): number {
  // Count velocity sign changes in sliding window
  let transitions = 0;
  for (let i = 1; i < velocityHistory.length; i++) {
    const wasMoving = velocityHistory[i-1] > STOP_THRESHOLD;
    const isMoving = velocityHistory[i] > STOP_THRESHOLD;
    if (wasMoving !== isMoving) transitions++;
  }
  return transitions / (velocityHistory.length / 60);  // per minute
}
```

**Pattern: Opposing Flows**

Simulated by:
- Spawning agents with opposite target directions
- Lane formation emerges from collision avoidance
- Turbulence at boundaries indicates risk

Detection via directional divergence:
```typescript
function computeDirectionalDivergence(agents: Agent[]): number {
  if (agents.length < 2) return 0;
  
  const directions = agents.map(a => Math.atan2(a.velocity.y, a.velocity.x));
  
  // Compute circular variance
  let sumCos = 0, sumSin = 0;
  for (const dir of directions) {
    sumCos += Math.cos(dir);
    sumSin += Math.sin(dir);
  }
  
  const R = Math.sqrt(sumCos*sumCos + sumSin*sumSin) / agents.length;
  return 1 - R;  // 0 = uniform, 1 = maximally divergent
}
```

### 3.3 Sensor Data Simulation

Each choke point samples agents within its region and computes synthetic sensor readings.

**mmWave Simulation**

```typescript
function simulateMmWaveReading(
  chokePoint: ChokePoint,
  agents: Agent[],
  prevReading: MmWaveSensorData | null
): MmWaveSensorData {
  
  const agentsInRegion = getAgentsInChokePoint(agents, chokePoint);
  
  if (agentsInRegion.length === 0) {
    return {
      timestamp: Date.now(),
      chokePointId: chokePoint.id,
      avgVelocity: 0,
      velocityVariance: 0,
      dominantDirection: 0,
      directionalDivergence: 0,
      stopGoFrequency: 0,
      stationaryRatio: 1
    };
  }
  
  // Compute velocity statistics
  const speeds = agentsInRegion.map(a => Vector.magnitude(a.velocity));
  const avgVelocity = mean(speeds);
  const velocityVariance = variance(speeds);
  
  // Direction analysis
  const directions = agentsInRegion.map(a => 
    Math.atan2(a.velocity.y, a.velocity.x)
  );
  const dominantDirection = circularMean(directions);
  const directionalDivergence = computeDirectionalDivergence(agentsInRegion);
  
  // Stationary detection
  const stationaryCount = agentsInRegion.filter(
    a => Vector.magnitude(a.velocity) < STOP_THRESHOLD
  ).length;
  const stationaryRatio = stationaryCount / agentsInRegion.length;
  
  // Stop-go requires history (use buffer)
  const stopGoFrequency = computeStopGoFromHistory(chokePoint.id);
  
  // Add realistic noise
  return {
    timestamp: Date.now(),
    chokePointId: chokePoint.id,
    avgVelocity: addNoise(avgVelocity, 0.05),
    velocityVariance: addNoise(velocityVariance, 0.1),
    dominantDirection,
    directionalDivergence: addNoise(directionalDivergence, 0.05),
    stopGoFrequency,
    stationaryRatio
  };
}
```

**Audio Simulation**

```typescript
function simulateAudioReading(
  chokePoint: ChokePoint,
  agents: Agent[]
): AudioSensorData {
  
  const agentsInRegion = getAgentsInChokePoint(agents, chokePoint);
  const density = agentsInRegion.length / getChokePointArea(chokePoint);
  
  // Base sound level correlates with density
  let soundLevel = Math.min(1, density / MAX_DENSITY);
  
  // Add variability based on agent states
  const distressedRatio = agentsInRegion.filter(
    a => a.state === 'pushing' || a.patience < 0.3
  ).length / Math.max(1, agentsInRegion.length);
  
  soundLevel += distressedRatio * 0.3;
  
  // Spike detection (stochastic, more likely when distressed)
  const spikeProb = distressedRatio * 0.1;  // 10% chance when fully distressed
  const spikeDetected = Math.random() < spikeProb;
  
  return {
    timestamp: Date.now(),
    chokePointId: chokePoint.id,
    soundEnergyLevel: clamp(addNoise(soundLevel, 0.1), 0, 1),
    spikeDetected,
    spikeIntensity: spikeDetected ? 0.5 + Math.random() * 0.5 : 0,
    audioCharacter: soundLevel > 0.7 ? 'distressed' : 
                    soundLevel > 0.4 ? 'loud' : 'ambient'
  };
}
```

**Camera Simulation**

```typescript
function simulateCameraReading(
  chokePoint: ChokePoint,
  agents: Agent[]
): CameraSensorData {
  
  const agentsInRegion = getAgentsInChokePoint(agents, chokePoint);
  const area = getChokePointArea(chokePoint);
  
  const crowdDensity = agentsInRegion.length / area;
  
  // Occlusion increases with density (simulating camera view blockage)
  const occlusionPercentage = Math.min(0.95, crowdDensity / 6);  // ~6 p/m² is crush
  
  // Optical flow consistency from velocity alignment
  const opticalFlowConsistency = 1 - computeDirectionalDivergence(agentsInRegion);
  
  return {
    timestamp: Date.now(),
    chokePointId: chokePoint.id,
    crowdDensity: addNoise(crowdDensity, 0.2),
    occlusionPercentage,
    opticalFlowConsistency
  };
}
```

### 3.4 Crowd Instability Index (CII) Computation

The CII is computed per choke point using a weighted, explainable formula.

**Formula**

```
CII = w₁·f(velocityVariance) 
    + w₂·f(stopGoFrequency) 
    + w₃·f(directionalDivergence)
    + w₄·f(densityRisk)
    + w₅·audioModifier

Where:
- f(x) normalizes each metric to [0, 1]
- Weights sum to 1.0
- audioModifier is multiplicative, not additive (per requirements)
```

**Pseudocode**

```typescript
interface CIIWeights {
  velocityVariance: number;     // w₁ = 0.25
  stopGo: number;               // w₂ = 0.25
  directional: number;          // w₃ = 0.25
  density: number;              // w₄ = 0.25
}

const DEFAULT_WEIGHTS: CIIWeights = {
  velocityVariance: 0.25,
  stopGo: 0.25,
  directional: 0.25,
  density: 0.25
};

function computeCII(
  sensorData: AggregatedSensorData,
  weights: CIIWeights = DEFAULT_WEIGHTS
): number {
  
  const mmwave = sensorData.mmwave;
  const camera = sensorData.camera;
  const audio = sensorData.audio;
  
  if (!mmwave) return 0;  // mmWave is primary, required
  
  // Normalize velocity variance (0.5 m²/s² is high variance)
  const velVarNorm = Math.min(1, mmwave.velocityVariance / 0.5);
  
  // Normalize stop-go (10 events/min is high)
  const stopGoNorm = Math.min(1, mmwave.stopGoFrequency / 10);
  
  // Directional divergence already in [0, 1]
  const dirNorm = mmwave.directionalDivergence;
  
  // Density risk (4 p/m² is uncomfortable, 6 is dangerous)
  const densityNorm = camera 
    ? Math.min(1, camera.crowdDensity / 5)
    : mmwave.stationaryRatio;  // fallback proxy
  
  // Base CII computation
  let cii = 
    weights.velocityVariance * velVarNorm +
    weights.stopGo * stopGoNorm +
    weights.directional * dirNorm +
    weights.density * densityNorm;
  
  // Audio modifier: amplifies CII if distress detected, but cannot trigger alone
  if (audio && audio.audioCharacter === 'distressed') {
    cii = cii * (1 + 0.2 * audio.soundEnergyLevel);  // up to 20% boost
  }
  if (audio && audio.spikeDetected) {
    cii = cii * (1 + 0.1 * audio.spikeIntensity);    // up to 10% boost
  }
  
  return clamp(cii, 0, 1);
}
```

**Explainability Output**

```typescript
interface CIIExplanation {
  cii: number;
  contributions: {
    factor: string;
    rawValue: number;
    normalizedValue: number;
    weight: number;
    contribution: number;
  }[];
  audioModifier: number;
  interpretation: string;
}

function explainCII(sensorData: AggregatedSensorData): CIIExplanation {
  // Returns breakdown for dashboard display
  // Shows judges exactly why CII is at its current level
}
```

### 3.5 Risk State Transitions

CII alone doesn't determine color - we use hysteresis to prevent flickering:

```typescript
type RiskLevel = 'green' | 'yellow' | 'red';

interface RiskThresholds {
  yellowEnter: number;   // 0.35 - enter yellow from green
  yellowExit: number;    // 0.25 - exit yellow to green
  redEnter: number;      // 0.65 - enter red from yellow
  redExit: number;       // 0.55 - exit red to yellow
}

function computeRiskLevel(
  currentLevel: RiskLevel,
  cii: number,
  thresholds: RiskThresholds
): RiskLevel {
  
  switch (currentLevel) {
    case 'green':
      if (cii >= thresholds.yellowEnter) return 'yellow';
      return 'green';
      
    case 'yellow':
      if (cii >= thresholds.redEnter) return 'red';
      if (cii < thresholds.yellowExit) return 'green';
      return 'yellow';
      
    case 'red':
      if (cii < thresholds.redExit) return 'yellow';
      return 'red';
  }
}
```

---

## 4. Visualization Strategy

### 4.1 Map Layer Stack

```
Layer 5: UI Overlays (tooltips, labels)     ← React DOM
Layer 4: Choke Point Regions (colored)      ← Leaflet Circle/Polygon
Layer 3: Heat Overlay (risk gradient)       ← Canvas overlay
Layer 2: Agent Particles                    ← Canvas overlay
Layer 1: Road/Path highlights               ← GeoJSON layer
Layer 0: OSM Tiles                          ← Leaflet TileLayer
```

### 4.2 Agent Rendering

Use HTML5 Canvas for performance (1000+ agents at 60fps):

```typescript
function renderAgents(ctx: CanvasRenderingContext2D, agents: Agent[]): void {
  for (const agent of agents) {
    const screenPos = geoToScreen(agent.position);
    
    // Color by state
    ctx.fillStyle = getAgentColor(agent.state);
    
    // Draw as small circle
    ctx.beginPath();
    ctx.arc(screenPos.x, screenPos.y, 3, 0, Math.PI * 2);
    ctx.fill();
    
    // Optional: velocity vector for debugging
    if (DEBUG_MODE) {
      ctx.strokeStyle = '#00ff00';
      ctx.beginPath();
      ctx.moveTo(screenPos.x, screenPos.y);
      ctx.lineTo(
        screenPos.x + agent.velocity.x * 10,
        screenPos.y + agent.velocity.y * 10
      );
      ctx.stroke();
    }
  }
}

function getAgentColor(state: AgentState): string {
  switch (state) {
    case 'moving': return '#4a90d9';   // blue
    case 'slowing': return '#f5a623';  // orange
    case 'stopped': return '#d0021b';  // red
    case 'pushing': return '#9013fe';  // purple
  }
}
```

### 4.3 Choke Point Visualization

```typescript
function renderChokePoint(
  chokePoint: ChokePoint,
  cii: number,
  map: L.Map
): L.Layer {
  
  const color = ciiToColor(cii);
  const opacity = 0.3 + cii * 0.4;  // more opaque when dangerous
  
  if (chokePoint.geometry.type === 'circle') {
    return L.circle(chokePoint.center, {
      radius: chokePoint.geometry.radius,
      color: color,
      fillColor: color,
      fillOpacity: opacity,
      weight: 2
    });
  } else {
    return L.polygon(chokePoint.geometry.vertices, {
      color: color,
      fillColor: color,
      fillOpacity: opacity,
      weight: 2
    });
  }
}

function ciiToColor(cii: number): string {
  // Smooth gradient from green → yellow → red
  if (cii < 0.35) {
    // Green to yellow
    const t = cii / 0.35;
    return lerpColor('#22c55e', '#eab308', t);
  } else {
    // Yellow to red
    const t = (cii - 0.35) / 0.65;
    return lerpColor('#eab308', '#dc2626', t);
  }
}
```

### 4.4 Dashboard Components

```
┌─────────────────────────────────────────────────────────────────┐
│  SIMULATION CONTROLS                                             │
├─────────────────────────────────────────────────────────────────┤
│  [▶ Play] [⏸ Pause] [⏹ Reset]  Speed: [1x ▼]                    │
│                                                                  │
│  Crowd Inflow: ████████░░ 80%    [+] [-]                        │
│                                                                  │
│  Perturbations:                                                  │
│  [x] Opposing Flow   [ ] Block Exit A   [ ] Density Surge       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  CHOKE POINT: Gate A                              CII: 0.72 🔴  │
├─────────────────────────────────────────────────────────────────┤
│  Sensors: [mmWave ✓] [Audio ✓] [Camera ✓]  Confidence: 94%     │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Velocity Variance                                          ││
│  │  ▁▂▃▄▅▆▇█▇▆▅▄▃▄▅▆▇█████▇▆▅▄▅▆▇████████  (30s window)       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Stop-Go Frequency                                          ││
│  │  ▁▁▂▂▃▃▄▄▅▅▆▆▇▇████████████▇▇▆▆▅▅▄▄▃▃  (30s window)         ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  CII Breakdown:                                                  │
│  Vel. Variance:  ████████░░ 0.82 × 0.25 = 0.205                 │
│  Stop-Go:        ███████░░░ 0.71 × 0.25 = 0.178                 │
│  Dir. Divergence:█████░░░░░ 0.54 × 0.25 = 0.135                 │
│  Density:        ████████░░ 0.79 × 0.25 = 0.198                 │
│  Audio Modifier: ×1.08                                           │
│  ────────────────────────────────────────                        │
│  Total CII:      0.72                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Implementation Plan

### Phase 1: Foundation (Day 1 - 4 hours)

**Goal**: Basic map with OSM tiles and project structure

1. Initialize React + TypeScript project with Vite
2. Install dependencies:
   - `react-leaflet`, `leaflet`
   - `zustand` for state
   - `recharts` for graphs
3. Create basic map component showing a fixed location
4. Implement coordinate conversion utilities (geo ↔ pixel)
5. Add Zustand store skeleton

**Deliverable**: Map displaying OSM tiles, centered on demo location

### Phase 2: Choke Points (Day 1 - 3 hours)

**Goal**: Manual choke point placement

1. Implement click-to-place choke point
2. Add circle/polygon drawing modes
3. Create choke point list panel
4. Store choke points in Zustand
5. Render choke points on map

**Deliverable**: Can place and visualize choke points

### Phase 3: Agent System (Day 2 - 5 hours)

**Goal**: Crowd particle simulation

1. Implement Vector utility class
2. Create Agent spawn/despawn logic
3. Implement path-following waypoints (manual definition for demo)
4. Add steering behaviors (separation, collision avoidance)
5. Canvas overlay for agent rendering
6. 60fps update loop with requestAnimationFrame

**Deliverable**: Agents flowing through defined paths

### Phase 4: Sensor Simulation (Day 2 - 4 hours)

**Goal**: Synthetic sensor data generation

1. Implement spatial queries (agents in choke point)
2. Create mmWave sensor simulator
3. Create audio sensor simulator
4. Create camera sensor simulator
5. Add time-series buffers for history
6. 10Hz sensor sampling loop

**Deliverable**: Sensor data streams per choke point

### Phase 5: CII Engine (Day 3 - 3 hours)

**Goal**: Risk computation

1. Implement CII formula
2. Add hysteresis for state transitions
3. Create CII explainability output
4. Connect CII to choke point color updates

**Deliverable**: Choke points change color based on CII

### Phase 6: Dashboard (Day 3 - 4 hours)

**Goal**: Control panel and metrics display

1. Simulation controls (play/pause/speed)
2. Perturbation triggers
3. Per-choke-point metrics panel
4. Time-series graphs with Recharts
5. CII breakdown display

**Deliverable**: Full control and visibility

### Phase 7: Demo Scenarios (Day 4 - 3 hours)

**Goal**: Preset scenarios for demo

1. "Normal Flow" scenario
2. "Opposing Flow" scenario
3. "Exit Blockage" scenario
4. "Density Surge" scenario
5. Smooth transitions between scenarios

**Deliverable**: One-click scenario triggers

### Phase 8: Polish (Day 4 - 3 hours)

**Goal**: Demo-ready quality

1. Visual polish (colors, transitions, animations)
2. Performance optimization (spatial hashing)
3. Mobile/projection-friendly layout
4. Edge case handling
5. Loading states

**Deliverable**: Demo-ready application

---

## 6. Technical Assumptions & Constraints

### Assumptions

1. **Demo location**: A single, pre-selected area (e.g., temple complex entrance)
2. **Agent count**: 200-500 agents for smooth performance
3. **Choke points**: 3-5 per demo scenario
4. **Real-time simulation**: Not historical playback
5. **Browser-only**: No backend required

### Constraints

1. **No actual sensor data**: All data is simulated
2. **Simplified physics**: No true fluid dynamics
3. **2D only**: No height/elevation modeling
4. **Pre-defined paths**: Agents follow manual waypoints, not OSM routing

### Performance Targets

| Metric | Target |
|--------|--------|
| Frame rate | 60 FPS sustained |
| Agent count | 500 without degradation |
| Memory | < 200 MB |
| Initial load | < 3 seconds |

---

## 7. File Structure

```
/src
  /components
    /map
      MapView.tsx           # Main map container
      ChokePointLayer.tsx   # Choke point rendering
      AgentCanvas.tsx       # Canvas overlay for agents
      HeatOverlay.tsx       # Risk heat visualization
    /dashboard
      ControlPanel.tsx      # Sim controls
      ChokePointCard.tsx    # Per-CP metrics
      TimeSeriesChart.tsx   # Recharts wrapper
      CIIBreakdown.tsx      # Explainability display
    /common
      Button.tsx
      Slider.tsx
      Toggle.tsx
  
  /simulation
    engine.ts               # Main simulation loop
    agents.ts               # Agent update logic
    steering.ts             # Steering behaviors
    sensors.ts              # Sensor simulators
    cii.ts                  # CII computation
  
  /store
    index.ts                # Zustand store
    types.ts                # TypeScript interfaces
  
  /utils
    vector.ts               # 2D vector math
    geo.ts                  # Coordinate conversion
    math.ts                 # Statistical functions
    colors.ts               # Color interpolation
  
  /data
    demoLocation.ts         # Pre-configured demo area
    scenarios.ts            # Preset scenarios
  
  App.tsx
  main.tsx
  index.css
```

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Performance issues with many agents | Spatial hashing, canvas batching, reduce agent count |
| CII not responding as expected | Tunable weights, real-time debug overlay |
| Map tiles not loading | Fallback to local cache, simple grid background |
| Judges don't understand CII | Prominent explainability panel, color legend |
| Demo machine issues | Build as static site, host on GitHub Pages backup |

---

## 9. Success Criteria

The demo is successful if:

1. **Judges can see** crowd flow on a real map
2. **Judges can understand** what each choke point represents
3. **CII rises BEFORE** visible chaos (early warning demo)
4. **CII explanation** is clear and believable
5. **Perturbations** (block exit, opposing flow) cause visible CII increase
6. **System looks professional**, not like a game or toy

---

*End of Technical Design Document*
