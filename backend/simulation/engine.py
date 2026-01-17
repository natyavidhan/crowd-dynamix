"""
Main simulation engine.
Coordinates road-constrained agents, sensors, and CII computation.
"""

import asyncio
import time
from typing import List, Dict, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector import vec2, Vec2, from_angle, magnitude
from models.types import (
    ChokePoint, SimulationConfig, SimulationState, AgentSnapshot,
    AggregatedSensorData, CIIExplanation, GeoPoint, SimPoint,
    SensorConfig, SensorType, CircleGeometry, RiskState
)

# New road-constrained movement system
from simulation.road_movement import (
    RoadAgentPool, RoadNetworkManager, RoadSegment, RoadSpawnPoint,
    update_road_agents, spawn_road_agents
)

from simulation.sensors import SensorBuffers, simulate_all_sensors
from simulation.cii import update_risk_state, DEFAULT_WEIGHTS

# Import venue loader
from configs.venues import VenueConfig, VenueLoader, list_available_venues, load_venue


# ============================================================================
# Coordinate Conversion
# ============================================================================

@dataclass
class CoordinateSystem:
    """Handles geo <-> simulation coordinate conversion."""
    
    origin: GeoPoint
    meters_per_deg_lat: float = 111320.0
    meters_per_deg_lng: float = 111320.0
    
    def __post_init__(self):
        lat_rad = np.radians(self.origin.lat)
        self.meters_per_deg_lng = 111320.0 * np.cos(lat_rad)
    
    def geo_to_sim(self, geo: GeoPoint) -> SimPoint:
        x = (geo.lng - self.origin.lng) * self.meters_per_deg_lng
        y = (geo.lat - self.origin.lat) * self.meters_per_deg_lat
        return SimPoint(x=x, y=y)
    
    def sim_to_geo(self, sim: SimPoint) -> GeoPoint:
        lng = self.origin.lng + sim.x / self.meters_per_deg_lng
        lat = self.origin.lat + sim.y / self.meters_per_deg_lat
        return GeoPoint(lat=lat, lng=lng)


# ============================================================================
# Venue Data Container
# ============================================================================

@dataclass
class VenueInfo:
    """Current venue information for API responses."""
    id: str
    name: str
    location: str
    description: str
    roads: List[Dict]
    spawn_point_count: int
    choke_point_count: int
    origin: Dict


# ============================================================================
# Agent Pool Adapter (for sensor compatibility)
# ============================================================================

class AgentPoolAdapter:
    """
    Adapts RoadAgentPool to match the interface expected by sensors.
    Provides positions, velocities, states arrays.
    """
    def __init__(self, road_pool: RoadAgentPool):
        self.road_pool = road_pool
        self._update_cache()
    
    def _update_cache(self):
        """Update cached arrays from road agents."""
        agents = list(self.road_pool.agents.values())
        n = len(agents)
        
        if n == 0:
            self._positions = np.zeros((0, 2))
            self._velocities = np.zeros((0, 2))
            self._states = np.zeros(0, dtype=np.int32)
            self._ids = np.zeros(0, dtype=np.int32)
            self._active = np.zeros(0, dtype=bool)
            self.count = 0
            return
        
        self._positions = np.zeros((n, 2))
        self._velocities = np.zeros((n, 2))
        self._states = np.zeros(n, dtype=np.int32)
        self._ids = np.zeros(n, dtype=np.int32)
        self._active = np.ones(n, dtype=bool)
        
        for i, agent in enumerate(agents):
            pos = agent.get_world_position(self.road_pool.network.roads)
            self._positions[i] = [pos[0], pos[1]]
            
            # Compute velocity from speed and road direction
            road = self.road_pool.network.roads.get(agent.road_id)
            if road:
                direction = road.direction_at_distance(agent.distance_along)
                vel = direction * agent.speed * agent.direction
                self._velocities[i] = [vel[0], vel[1]]
            
            # State: 0=moving, 1=slowing, 2=stopped
            if agent.waiting or agent.speed < 0.1:
                self._states[i] = 2  # stopped
            elif agent.speed < agent.target_speed * 0.5:
                self._states[i] = 1  # slowing
            else:
                self._states[i] = 0  # moving
            
            self._ids[i] = agent.id
        
        self.count = n
    
    @property
    def positions(self):
        return self._positions
    
    @property
    def velocities(self):
        return self._velocities
    
    @property
    def states(self):
        return self._states
    
    @property
    def ids(self):
        return self._ids
    
    @property
    def active(self):
        return self._active
    
    def get_active_indices(self):
        return np.arange(self.count)
    
    def get_positions(self):
        return self._positions


# ============================================================================
# Simulation Engine
# ============================================================================

@dataclass
class SimulationEngine:
    """Main simulation engine using road-constrained movement."""
    
    config: SimulationConfig = field(default_factory=SimulationConfig)
    coord_sys: CoordinateSystem = field(default_factory=lambda: CoordinateSystem(
        origin=GeoPoint(lat=12.9716, lng=77.5946)
    ))
    
    configs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "configs" / "venues")
    
    current_venue: Optional[VenueConfig] = None
    current_venue_id: Optional[str] = None
    venue_loader: Optional[VenueLoader] = None
    
    # Road-constrained agent system
    road_network: RoadNetworkManager = field(default_factory=RoadNetworkManager)
    road_agent_pool: Optional[RoadAgentPool] = None
    road_spawn_points: List[RoadSpawnPoint] = field(default_factory=list)
    
    # Adapter for sensor compatibility
    agent_pool_adapter: Optional[AgentPoolAdapter] = None
    
    # Legacy compatibility
    agent_pool: Optional[object] = None
    
    # Sensor system
    sensor_buffers: SensorBuffers = field(init=False)
    
    # State
    choke_points: List[ChokePoint] = field(default_factory=list)
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
        self.sensor_buffers = SensorBuffers()
        
        # Try to load default venue
        if not self._try_load_default_venue():
            self._setup_demo_scenario()
    
    def _try_load_default_venue(self) -> bool:
        """Try to load a default venue config."""
        venues = list_available_venues(self.configs_dir)
        if venues:
            first_venue = sorted(venues, key=lambda v: v['id'])[0]
            return self.load_venue(first_venue['id'])
        return False
    
    def _setup_demo_scenario(self):
        """Setup a simple demo scenario with roads."""
        # Create a simple road network
        self.road_network = RoadNetworkManager()
        
        # Main road going north
        main_road = RoadSegment(
            id="main_road",
            name="Main Road",
            width=6.0,
            speed_limit=1.4,
            bidirectional=True,
            centerline=[
                vec2(0, -50),
                vec2(0, 0),
                vec2(0, 50),
                vec2(0, 100)
            ]
        )
        self.road_network.add_road(main_road)
        
        # Add exit at end
        self.road_network.add_exit("exit_north", vec2(0, 100))
        
        # Build connections
        self.road_network.build_connections()
        
        # Create agent pool
        self.road_agent_pool = RoadAgentPool(
            max_agents=self.config.max_agents,
            network=self.road_network
        )
        
        # Create spawn points
        self.road_spawn_points = [
            RoadSpawnPoint(
                id="spawn_south",
                road_id="main_road",
                distance_along=0,
                target_exit_id="exit_north",
                rate=5.0
            )
        ]
        
        # Setup adapter
        self.agent_pool_adapter = AgentPoolAdapter(self.road_agent_pool)
        self.agent_pool = self.agent_pool_adapter
        
        # Create demo choke points
        self.choke_points = self._create_demo_choke_points()
    
    def _create_demo_choke_points(self) -> List[ChokePoint]:
        """Create default choke points for demo."""
        return [
            ChokePoint(
                id="cp_entrance",
                name="Main Entrance",
                center=self.coord_sys.sim_to_geo(SimPoint(x=0, y=-30)),
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
                center=self.coord_sys.sim_to_geo(SimPoint(x=0, y=0)),
                geometry=CircleGeometry(radius=20),
                sensors=[
                    SensorConfig(type=SensorType.MMWAVE, enabled=True),
                    SensorConfig(type=SensorType.AUDIO, enabled=True),
                    SensorConfig(type=SensorType.CAMERA, enabled=True),
                ],
                sim_center=SimPoint(x=0, y=0),
                sim_radius=20
            ),
        ]
    
    def get_available_venues(self) -> List[Dict]:
        """Get list of available venue configurations."""
        return list_available_venues(self.configs_dir)
    
    def load_venue(self, venue_id: str) -> bool:
        """Load a venue configuration by ID."""
        config = load_venue(self.configs_dir, venue_id)
        if config is None:
            return False
        
        self.current_venue = config
        self.current_venue_id = venue_id
        self.venue_loader = VenueLoader(config)
        
        # Update coordinate system with new origin
        self.coord_sys = CoordinateSystem(
            origin=GeoPoint(lat=config.origin.lat, lng=config.origin.lng)
        )
        
        # Build road network from venue config
        self._build_road_network_from_venue()
        
        # Load choke points
        self.choke_points = self.venue_loader.choke_points
        
        # Apply venue defaults to config
        if config.defaults:
            if config.defaults.agent_count:
                self.config.max_agents = config.defaults.agent_count
            if config.defaults.spawn_rate:
                self.config.base_inflow_rate = config.defaults.spawn_rate
        
        # Reset simulation
        self.reset()
        
        return True
    
    def _build_road_network_from_venue(self):
        """Build road network from venue loader."""
        if not self.venue_loader:
            return
        
        self.road_network = RoadNetworkManager()
        
        # Convert roads from venue loader
        for road in self.venue_loader.road_network.get_all_roads():
            road_segment = RoadSegment(
                id=road.id,
                name=road.name,
                width=road.width,
                speed_limit=road.speed_limit,
                bidirectional=road.bidirectional,
                centerline=road.centerline
            )
            self.road_network.add_road(road_segment)
        
        # Add exits
        for exit_id, exit_pos in self.venue_loader.exits.items():
            self.road_network.add_exit(exit_id, exit_pos)
        
        # Build connections
        self.road_network.build_connections()
        
        # Create spawn points on roads
        self.road_spawn_points = []
        for psp in self.venue_loader.spawn_points:
            # Find the road this spawn point is on
            result = self.road_network.find_nearest_road(psp.position)
            if result:
                road_id, dist_along, perp_dist = result
                
                # Use target exit from spawn point config directly
                target_exit_id = psp.target_exit_id
                
                # If no exit specified, try to find from waypoints
                if target_exit_id is None and psp.waypoints and len(psp.waypoints) > 1:
                    # Find exit nearest to last waypoint
                    last_wp = psp.waypoints[-1]
                    min_exit_dist = float('inf')
                    for exit_id, exit_pos in self.road_network.exits.items():
                        d = magnitude(last_wp - exit_pos)
                        if d < min_exit_dist:
                            min_exit_dist = d
                            target_exit_id = exit_id
                
                # If still no exit, use first available
                if target_exit_id is None and self.road_network.exits:
                    target_exit_id = list(self.road_network.exits.keys())[0]
                
                spawn_point = RoadSpawnPoint(
                    id=psp.id,
                    road_id=road_id,
                    distance_along=dist_along,
                    target_exit_id=target_exit_id,
                    rate=psp.rate
                )
                self.road_spawn_points.append(spawn_point)
        
        # Create agent pool
        self.road_agent_pool = RoadAgentPool(
            max_agents=self.config.max_agents,
            network=self.road_network
        )
        
        # Setup adapter
        self.agent_pool_adapter = AgentPoolAdapter(self.road_agent_pool)
        self.agent_pool = self.agent_pool_adapter
    
    def get_current_venue_info(self) -> Optional[VenueInfo]:
        """Get info about currently loaded venue."""
        if not self.current_venue or not self.venue_loader:
            return None
        
        return VenueInfo(
            id=self.current_venue_id,
            name=self.current_venue.venue.name,
            location=self.current_venue.venue.location,
            description=self.current_venue.venue.description or "",
            roads=self.venue_loader.get_road_geometries_geo(),
            spawn_point_count=len(self.road_spawn_points),
            choke_point_count=len(self.choke_points),
            origin={
                "lat": self.current_venue.origin.lat,
                "lng": self.current_venue.origin.lng
            }
        )
    
    def reset(self):
        """Reset simulation to initial state."""
        # Recreate road agent pool
        if self.road_network:
            self.road_agent_pool = RoadAgentPool(
                max_agents=self.config.max_agents,
                network=self.road_network
            )
            self.agent_pool_adapter = AgentPoolAdapter(self.road_agent_pool)
            self.agent_pool = self.agent_pool_adapter
        
        self.sensor_buffers = SensorBuffers()
        self.sensor_data.clear()
        self.cii_explanations.clear()
        self.tick = 0
        
        # Reset choke point risk states
        for cp in self.choke_points:
            cp.risk_state = RiskState()
    
    def set_origin(self, origin: GeoPoint):
        """Set the map origin point."""
        self.coord_sys = CoordinateSystem(origin=origin)
        self.reset()
    
    def add_choke_point(self, choke_point: ChokePoint):
        """Add a new choke point."""
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
    
    def step(self, dt: float):
        """Advance simulation by dt seconds."""
        if not self.config.running:
            return
        
        if not self.road_agent_pool:
            return
        
        # Apply speed multiplier
        effective_dt = dt * self.config.speed_multiplier
        
        # Spawn new agents
        inflow_rate = self.config.base_inflow_rate * self.config.current_inflow_multiplier
        
        if self.config.density_surge_enabled:
            inflow_rate *= 2.0
        
        # Spawn on roads
        spawn_road_agents(
            self.road_agent_pool,
            self.road_spawn_points,
            effective_dt,
            self.rng
        )
        
        # Update road-constrained agents
        update_road_agents(self.road_agent_pool, effective_dt)
        
        # Update adapter cache
        if self.agent_pool_adapter:
            self.agent_pool_adapter._update_cache()
        
        self.tick += 1
    
    def sample_sensors(self):
        """Sample sensor data for all choke points."""
        if not self.agent_pool_adapter:
            return
        
        for cp in self.choke_points:
            sensor_data = simulate_all_sensors(cp, self.agent_pool_adapter, self.sensor_buffers)
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
    
    _state_map = {0: "moving", 1: "slowing", 2: "stopped", 3: "pushing"}
    
    def get_state(self) -> SimulationState:
        """Get current simulation state for broadcasting."""
        agent_snapshots = []
        
        if self.road_agent_pool and self.agent_pool_adapter:
            positions = self.agent_pool_adapter.positions
            velocities = self.agent_pool_adapter.velocities
            states = self.agent_pool_adapter.states
            ids = self.agent_pool_adapter.ids
            
            for i in range(self.agent_pool_adapter.count):
                agent_snapshots.append(AgentSnapshot(
                    id=int(ids[i]),
                    x=float(positions[i, 0]),
                    y=float(positions[i, 1]),
                    vx=float(velocities[i, 0]),
                    vy=float(velocities[i, 1]),
                    state=self._state_map.get(states[i], "moving")
                ))
        
        origin = GeoPoint(
            lat=self.coord_sys.origin.lat,
            lng=self.coord_sys.origin.lng
        )
        
        return SimulationState(
            tick=self.tick,
            agents=agent_snapshots,
            choke_points=self.choke_points,
            sensor_data=self.sensor_data,
            cii_explanations=self.cii_explanations,
            config=self.config,
            origin=origin
        )
    
    async def run_loop(
        self,
        target_fps: float = 60.0,
        sensor_hz: float = 10.0,
        cii_hz: float = 1.0,
        broadcast_hz: float = 15.0
    ):
        """Main simulation loop."""
        frame_time = 1.0 / target_fps
        sensor_interval = 1.0 / sensor_hz
        cii_interval = 1.0 / cii_hz
        broadcast_interval = 1.0 / broadcast_hz
        
        last_broadcast = time.time()
        
        while True:
            loop_start = time.time()
            
            now = time.time()
            dt = now - self.last_update
            self.last_update = now
            
            if self.config.running:
                self.step(dt)
            
            if now - self.last_sensor_sample >= sensor_interval:
                self.sample_sensors()
                self.last_sensor_sample = now
            
            if now - self.last_cii_compute >= cii_interval:
                self.compute_cii()
                self.last_cii_compute = now
            
            if now - last_broadcast >= broadcast_interval:
                if self.on_state_update:
                    state = self.get_state()
                    await self.on_state_update(state)
                last_broadcast = now
            
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_time - elapsed)
            await asyncio.sleep(sleep_time)
