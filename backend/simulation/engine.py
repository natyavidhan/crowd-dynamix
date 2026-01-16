"""
Main simulation engine.
Coordinates agents, sensors, and CII computation.
Runs the simulation loop and broadcasts state.
"""

import asyncio
import time
from typing import List, Dict, Optional, Callable, Awaitable
from dataclasses import dataclass, field
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector import vec2, Vec2, from_angle
from models.types import (
    ChokePoint, SimulationConfig, SimulationState, AgentSnapshot,
    AggregatedSensorData, CIIExplanation, GeoPoint, SimPoint,
    SensorConfig, SensorType, CircleGeometry, RiskState
)
from simulation.agents import (
    AgentPool, SpatialHash, SpawnPoint, 
    update_agents, spawn_agents_at_rate
)
from simulation.sensors import SensorBuffers, simulate_all_sensors
from simulation.cii import update_risk_state, DEFAULT_WEIGHTS


# ============================================================================
# Coordinate Conversion
# ============================================================================

@dataclass
class CoordinateSystem:
    """Handles geo <-> simulation coordinate conversion."""
    
    origin: GeoPoint  # Reference point (becomes 0,0 in sim coords)
    meters_per_deg_lat: float = 111320.0
    meters_per_deg_lng: float = 111320.0  # Adjusted by latitude
    
    def __post_init__(self):
        # Adjust longitude scale by latitude
        lat_rad = np.radians(self.origin.lat)
        self.meters_per_deg_lng = 111320.0 * np.cos(lat_rad)
    
    def geo_to_sim(self, geo: GeoPoint) -> SimPoint:
        """Convert geographic to simulation coordinates."""
        x = (geo.lng - self.origin.lng) * self.meters_per_deg_lng
        y = (geo.lat - self.origin.lat) * self.meters_per_deg_lat
        return SimPoint(x=x, y=y)
    
    def sim_to_geo(self, sim: SimPoint) -> GeoPoint:
        """Convert simulation to geographic coordinates."""
        lng = self.origin.lng + sim.x / self.meters_per_deg_lng
        lat = self.origin.lat + sim.y / self.meters_per_deg_lat
        return GeoPoint(lat=lat, lng=lng)


# ============================================================================
# Demo Scenario Configuration
# ============================================================================

def create_demo_spawn_points(
    coord_sys: CoordinateSystem,
    opposing_flow: bool = False
) -> List[SpawnPoint]:
    """Create spawn points for demo scenario."""
    
    # Main flow: spawn from bottom, move to top
    main_spawn = vec2(0, -50)  # 50m south of origin
    main_waypoints = [
        vec2(0, 0),     # through center
        vec2(0, 50),    # to north
        vec2(0, 100),   # exit
    ]
    
    spawn_points = [
        SpawnPoint(
            position=main_spawn,
            direction=np.pi / 2,  # north
            waypoints=main_waypoints
        ),
        # Second spawn slightly offset
        SpawnPoint(
            position=vec2(-10, -50),
            direction=np.pi / 2,
            waypoints=[vec2(-5, 0), vec2(0, 50), vec2(5, 100)]
        ),
        SpawnPoint(
            position=vec2(10, -50),
            direction=np.pi / 2,
            waypoints=[vec2(5, 0), vec2(0, 50), vec2(-5, 100)]
        ),
    ]
    
    if opposing_flow:
        # Add counter-flow from top
        opposing_spawn = vec2(0, 100)
        opposing_waypoints = [
            vec2(0, 50),
            vec2(0, 0),
            vec2(0, -50),
        ]
        spawn_points.append(SpawnPoint(
            position=opposing_spawn,
            direction=-np.pi / 2,  # south
            waypoints=opposing_waypoints
        ))
    
    return spawn_points


def create_demo_choke_points(coord_sys: CoordinateSystem) -> List[ChokePoint]:
    """Create default choke points for demo."""
    
    return [
        ChokePoint(
            id="cp_entrance",
            name="Main Entrance",
            center=coord_sys.sim_to_geo(SimPoint(x=0, y=-30)),
            geometry=CircleGeometry(radius=15),
            sensors=[
                SensorConfig(type=SensorType.MMWAVE, enabled=True),
                SensorConfig(type=SensorType.AUDIO, enabled=True),
                SensorConfig(type=SensorType.CAMERA, enabled=True),
            ],
            sim_center=SimPoint(x=0, y=-30),
            sim_radius=15
        ),
        ChokePoint(
            id="cp_junction",
            name="Central Junction",
            center=coord_sys.sim_to_geo(SimPoint(x=0, y=0)),
            geometry=CircleGeometry(radius=20),
            sensors=[
                SensorConfig(type=SensorType.MMWAVE, enabled=True),
                SensorConfig(type=SensorType.AUDIO, enabled=True),
                SensorConfig(type=SensorType.CAMERA, enabled=True),
            ],
            sim_center=SimPoint(x=0, y=0),
            sim_radius=20
        ),
        ChokePoint(
            id="cp_exit",
            name="North Exit",
            center=coord_sys.sim_to_geo(SimPoint(x=0, y=50)),
            geometry=CircleGeometry(radius=15),
            sensors=[
                SensorConfig(type=SensorType.MMWAVE, enabled=True),
                SensorConfig(type=SensorType.AUDIO, enabled=True),
                SensorConfig(type=SensorType.CAMERA, enabled=True),
            ],
            sim_center=SimPoint(x=0, y=50),
            sim_radius=15
        ),
    ]


# ============================================================================
# Simulation Engine
# ============================================================================

@dataclass
class SimulationEngine:
    """Main simulation engine that coordinates all subsystems."""
    
    # Configuration
    config: SimulationConfig = field(default_factory=SimulationConfig)
    coord_sys: CoordinateSystem = field(default_factory=lambda: CoordinateSystem(
        origin=GeoPoint(lat=12.9716, lng=77.5946)  # Default: Bangalore
    ))
    
    # Subsystems
    agent_pool: AgentPool = field(init=False)
    spatial_hash: SpatialHash = field(init=False)
    sensor_buffers: SensorBuffers = field(init=False)
    
    # State
    choke_points: List[ChokePoint] = field(default_factory=list)
    spawn_points: List[SpawnPoint] = field(default_factory=list)
    sensor_data: Dict[str, AggregatedSensorData] = field(default_factory=dict)
    cii_explanations: Dict[str, CIIExplanation] = field(default_factory=dict)
    
    # Timing
    tick: int = 0
    last_update: float = field(default_factory=time.time)
    last_sensor_sample: float = field(default_factory=time.time)
    last_cii_compute: float = field(default_factory=time.time)
    
    # Random state
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    
    # Callbacks
    on_state_update: Optional[Callable[[SimulationState], Awaitable[None]]] = None
    
    def __post_init__(self):
        self.agent_pool = AgentPool(max_agents=self.config.max_agents)
        self.spatial_hash = SpatialHash(cell_size=3.0)
        self.sensor_buffers = SensorBuffers()
        
        # Initialize demo scenario
        self.spawn_points = create_demo_spawn_points(self.coord_sys)
        self.choke_points = create_demo_choke_points(self.coord_sys)
    
    def reset(self):
        """Reset simulation to initial state."""
        self.agent_pool = AgentPool(max_agents=self.config.max_agents)
        self.sensor_buffers = SensorBuffers()
        self.sensor_data.clear()
        self.cii_explanations.clear()
        self.tick = 0
        
        # Reset choke point risk states
        for cp in self.choke_points:
            cp.risk_state = RiskState()
        
        # Recreate spawn points based on current config
        self.spawn_points = create_demo_spawn_points(
            self.coord_sys,
            opposing_flow=self.config.opposing_flow_enabled
        )
    
    def set_origin(self, origin: GeoPoint):
        """Set the map origin point."""
        self.coord_sys = CoordinateSystem(origin=origin)
        self.reset()
    
    def add_choke_point(self, choke_point: ChokePoint):
        """Add a new choke point."""
        # Convert geo to sim coordinates
        choke_point.sim_center = self.coord_sys.geo_to_sim(choke_point.center)
        if hasattr(choke_point.geometry, 'radius'):
            choke_point.sim_radius = choke_point.geometry.radius
        self.choke_points.append(choke_point)
    
    def remove_choke_point(self, choke_point_id: str):
        """Remove a choke point by ID."""
        self.choke_points = [cp for cp in self.choke_points if cp.id != choke_point_id]
        self.sensor_data.pop(choke_point_id, None)
        self.cii_explanations.pop(choke_point_id, None)
    
    def update_config(self, **kwargs):
        """Update simulation configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Handle special cases
        if 'opposing_flow_enabled' in kwargs:
            self.spawn_points = create_demo_spawn_points(
                self.coord_sys,
                opposing_flow=self.config.opposing_flow_enabled
            )
    
    def step(self, dt: float):
        """Advance simulation by dt seconds."""
        if not self.config.running:
            return
        
        # Apply speed multiplier
        effective_dt = dt * self.config.speed_multiplier
        
        # Spawn new agents
        inflow_rate = self.config.base_inflow_rate * self.config.current_inflow_multiplier
        
        # Density surge doubles inflow
        if self.config.density_surge_enabled:
            inflow_rate *= 2.0
        
        spawn_agents_at_rate(
            self.agent_pool,
            self.spawn_points,
            inflow_rate,
            effective_dt,
            self.rng
        )
        
        # Update agent physics
        update_agents(self.agent_pool, self.spatial_hash, effective_dt)
        
        self.tick += 1
    
    def sample_sensors(self):
        """Sample sensor data for all choke points."""
        for cp in self.choke_points:
            sensor_data = simulate_all_sensors(cp, self.agent_pool, self.sensor_buffers)
            self.sensor_data[cp.id] = sensor_data
    
    def compute_cii(self):
        """Compute CII for all choke points."""
        for cp in self.choke_points:
            sensor_data = self.sensor_data.get(cp.id)
            if sensor_data is None:
                continue
            
            new_state, explanation = update_risk_state(
                cp.id,
                sensor_data,
                cp.risk_state
            )
            
            cp.risk_state = new_state
            self.cii_explanations[cp.id] = explanation
    
    def get_state(self) -> SimulationState:
        """Get current simulation state for broadcasting."""
        # Convert agents to snapshots
        agent_snapshots = []
        for idx in self.agent_pool.get_active_indices():
            pos = self.agent_pool.positions[idx]
            vel = self.agent_pool.velocities[idx]
            state_int = self.agent_pool.states[idx]
            state_map = {0: "moving", 1: "slowing", 2: "stopped", 3: "pushing"}
            
            agent_snapshots.append(AgentSnapshot(
                id=int(self.agent_pool.ids[idx]),
                x=float(pos[0]),
                y=float(pos[1]),
                vx=float(vel[0]),
                vy=float(vel[1]),
                state=state_map.get(state_int, "moving")
            ))
        
        return SimulationState(
            tick=self.tick,
            agents=agent_snapshots,
            choke_points=self.choke_points,
            sensor_data=self.sensor_data,
            cii_explanations=self.cii_explanations,
            config=self.config
        )
    
    async def run_loop(
        self,
        target_fps: float = 60.0,
        sensor_hz: float = 10.0,
        cii_hz: float = 1.0,
        broadcast_hz: float = 30.0
    ):
        """
        Main simulation loop.
        Runs until config.running is set to False.
        """
        frame_time = 1.0 / target_fps
        sensor_interval = 1.0 / sensor_hz
        cii_interval = 1.0 / cii_hz
        broadcast_interval = 1.0 / broadcast_hz
        
        last_broadcast = time.time()
        
        while True:
            loop_start = time.time()
            
            # Calculate dt
            now = time.time()
            dt = now - self.last_update
            self.last_update = now
            
            # Simulation step
            if self.config.running:
                self.step(dt)
            
            # Sensor sampling (10 Hz)
            if now - self.last_sensor_sample >= sensor_interval:
                self.sample_sensors()
                self.last_sensor_sample = now
            
            # CII computation (1 Hz)
            if now - self.last_cii_compute >= cii_interval:
                self.compute_cii()
                self.last_cii_compute = now
            
            # Broadcast state (30 Hz)
            if now - last_broadcast >= broadcast_interval:
                if self.on_state_update:
                    state = self.get_state()
                    await self.on_state_update(state)
                last_broadcast = now
            
            # Frame timing
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_time - elapsed)
            await asyncio.sleep(sleep_time)
