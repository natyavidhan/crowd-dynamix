"""
Data models for the crowd simulation system.
Uses Pydantic for validation and serialization.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
import time


# ============================================================================
# Geographic Types
# ============================================================================

class GeoPoint(BaseModel):
    """Geographic coordinate (WGS84)."""
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class SimPoint(BaseModel):
    """Simulation coordinate in meters from origin."""
    x: float
    y: float
    
    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


class GeoBounds(BaseModel):
    """Geographic bounding box."""
    north: float
    south: float
    east: float
    west: float


# ============================================================================
# Choke Point Models
# ============================================================================

class SensorType(str, Enum):
    MMWAVE = "mmwave"
    AUDIO = "audio"
    CAMERA = "camera"


class SensorConfig(BaseModel):
    """Configuration for a sensor at a choke point."""
    type: SensorType
    enabled: bool = True
    confidence: float = Field(default=0.95, ge=0, le=1)


class CircleGeometry(BaseModel):
    """Circular choke point region."""
    type: Literal["circle"] = "circle"
    radius: float = Field(..., gt=0, description="Radius in meters")


class PolygonGeometry(BaseModel):
    """Polygonal choke point region."""
    type: Literal["polygon"] = "polygon"
    vertices: List[GeoPoint] = Field(..., min_length=3)


class RiskState(BaseModel):
    """Current risk assessment for a choke point."""
    cii: float = Field(default=0.0, ge=0, le=1)
    trend: Literal["stable", "rising", "falling"] = "stable"
    level: Literal["green", "yellow", "red"] = "green"
    last_updated: float = Field(default_factory=time.time)


class ChokePoint(BaseModel):
    """A monitored zone with sensors."""
    id: str
    name: str
    center: GeoPoint
    geometry: CircleGeometry | PolygonGeometry
    sensors: List[SensorConfig] = Field(default_factory=list)
    risk_state: RiskState = Field(default_factory=RiskState)
    
    # Simulation coordinates (computed from geo)
    sim_center: Optional[SimPoint] = None
    sim_radius: Optional[float] = None  # For circle geometry


# ============================================================================
# Agent Models
# ============================================================================

class AgentState(str, Enum):
    MOVING = "moving"
    SLOWING = "slowing"
    STOPPED = "stopped"
    PUSHING = "pushing"


class Agent(BaseModel):
    """A crowd particle/agent."""
    id: int
    position: SimPoint
    velocity: SimPoint
    target_velocity: SimPoint
    
    # Path following
    waypoints: List[SimPoint] = Field(default_factory=list)
    current_waypoint_idx: int = 0
    
    # Behavioral state
    state: AgentState = AgentState.MOVING
    patience: float = Field(default=1.0, ge=0, le=1)
    
    # Physical properties
    radius: float = Field(default=0.3, description="Collision radius in meters")
    max_speed: float = Field(default=1.4, description="Max speed in m/s")


class AgentSnapshot(BaseModel):
    """Lightweight agent data for frontend rendering."""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    state: AgentState


# ============================================================================
# Sensor Data Models
# ============================================================================

class MmWaveSensorData(BaseModel):
    """mmWave radar sensor output."""
    timestamp: float = Field(default_factory=time.time)
    choke_point_id: str
    
    # Velocity metrics
    avg_velocity: float = Field(ge=0, description="Average speed in m/s")
    velocity_variance: float = Field(ge=0)
    dominant_direction: float = Field(description="Radians, 0 = east")
    directional_divergence: float = Field(ge=0, le=1, description="0=uniform, 1=chaotic")
    
    # Stop-go detection
    stop_go_frequency: float = Field(ge=0, description="Events per minute")
    stationary_ratio: float = Field(ge=0, le=1)


class AudioSensorData(BaseModel):
    """Audio sensor output."""
    timestamp: float = Field(default_factory=time.time)
    choke_point_id: str
    
    sound_energy_level: float = Field(ge=0, le=1)
    spike_detected: bool = False
    spike_intensity: float = Field(default=0, ge=0, le=1)
    audio_character: Literal["ambient", "loud", "distressed"] = "ambient"


class CameraSensorData(BaseModel):
    """Camera/vision sensor output."""
    timestamp: float = Field(default_factory=time.time)
    choke_point_id: str
    
    crowd_density: float = Field(ge=0, description="People per m²")
    occlusion_percentage: float = Field(ge=0, le=1)
    optical_flow_consistency: float = Field(ge=0, le=1)


class AggregatedSensorData(BaseModel):
    """Combined sensor readings for a choke point."""
    choke_point_id: str
    timestamp: float = Field(default_factory=time.time)
    
    mmwave: Optional[MmWaveSensorData] = None
    audio: Optional[AudioSensorData] = None
    camera: Optional[CameraSensorData] = None
    
    overall_confidence: float = Field(default=1.0, ge=0, le=1)


# ============================================================================
# CII Models
# ============================================================================

class CIIContribution(BaseModel):
    """Single factor contribution to CII."""
    factor: str
    raw_value: float
    normalized_value: float
    weight: float
    contribution: float


class CIIExplanation(BaseModel):
    """Explainable breakdown of CII computation."""
    cii: float
    contributions: List[CIIContribution]
    audio_modifier: float = 1.0
    interpretation: str = ""


class CIIWeights(BaseModel):
    """Configurable weights for CII computation."""
    velocity_variance: float = 0.25
    stop_go: float = 0.25
    directional: float = 0.25
    density: float = 0.25


# ============================================================================
# Simulation Configuration
# ============================================================================

class SimulationConfig(BaseModel):
    """Simulation runtime configuration."""
    running: bool = False
    speed_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)
    
    # Crowd parameters
    base_inflow_rate: float = Field(default=5.0, description="Agents per second")
    current_inflow_multiplier: float = Field(default=1.0, ge=0, le=5)
    max_agents: int = Field(default=500)
    
    # Perturbations
    opposing_flow_enabled: bool = False
    blocked_exits: List[str] = Field(default_factory=list)
    density_surge_enabled: bool = False


class MapConfig(BaseModel):
    """Map and coordinate system configuration."""
    center: GeoPoint
    bounds: GeoBounds
    meters_per_degree_lat: float = 111320  # Approximate
    meters_per_degree_lng: float = 111320  # Adjusted by latitude


# ============================================================================
# WebSocket Message Types
# ============================================================================

class SimulationState(BaseModel):
    """Full simulation state broadcast to frontend."""
    timestamp: float = Field(default_factory=time.time)
    tick: int = 0
    
    agents: List[AgentSnapshot] = Field(default_factory=list)
    choke_points: List[ChokePoint] = Field(default_factory=list)
    sensor_data: Dict[str, AggregatedSensorData] = Field(default_factory=dict)
    cii_explanations: Dict[str, CIIExplanation] = Field(default_factory=dict)
    
    config: SimulationConfig = Field(default_factory=SimulationConfig)
    
    # Venue origin for coordinate conversion (sim coords -> geo coords)
    origin: Optional[GeoPoint] = None


class ControlMessage(BaseModel):
    """Control commands from frontend."""
    action: Literal[
        "start", "stop", "reset", 
        "set_speed", "set_inflow",
        "toggle_opposing_flow", "toggle_density_surge",
        "block_exit", "unblock_exit",
        "add_choke_point", "remove_choke_point"
    ]
    payload: Optional[Dict[str, Any]] = None
