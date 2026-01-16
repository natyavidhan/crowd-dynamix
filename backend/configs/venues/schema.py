"""
Venue configuration schema using Pydantic models.
Defines the structure for venue YAML/JSON files.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import yaml
from pathlib import Path


# ============================================================================
# Geographic Types
# ============================================================================

class GeoCoord(BaseModel):
    """Geographic coordinate."""
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


# ============================================================================
# Road Configuration
# ============================================================================

class RoadConfig(BaseModel):
    """Road/path configuration."""
    id: str
    name: str
    width: float = Field(default=6.0, gt=0, description="Road width in meters")
    centerline: List[GeoCoord] = Field(..., min_length=2, description="Road centerline points")
    bidirectional: bool = Field(default=True, description="Allow traffic both directions")
    speed_limit: float = Field(default=1.4, gt=0, description="Max speed in m/s")


# ============================================================================
# Spawn Point Configuration
# ============================================================================

class SpawnConfig(BaseModel):
    """Agent spawn point configuration."""
    id: str
    name: Optional[str] = None
    
    # Location - either on a road or at specific coordinates
    road_id: Optional[str] = Field(default=None, description="Road to spawn on")
    road_position: Optional[Literal["start", "end", "middle"]] = Field(default="start")
    location: Optional[GeoCoord] = Field(default=None, description="Specific spawn location")
    
    # Spawn parameters
    rate: float = Field(default=5.0, gt=0, description="Agents per second")
    spread: float = Field(default=3.0, ge=0, description="Random spread radius in meters")
    
    # Destination
    destination_road_id: Optional[str] = Field(default=None)
    destination_location: Optional[GeoCoord] = Field(default=None)
    exit_id: Optional[str] = Field(default=None, description="Exit point ID")
    
    # Flow direction (for opposing flow scenarios)
    flow_group: str = Field(default="main", description="Group for flow control")


# ============================================================================
# Choke Point Configuration
# ============================================================================

class ChokePointConfig(BaseModel):
    """Monitored zone configuration."""
    id: str
    name: str
    location: GeoCoord
    radius: float = Field(default=15.0, gt=0, description="Monitoring radius in meters")
    
    # Sensor configuration
    mmwave_enabled: bool = True
    audio_enabled: bool = True
    camera_enabled: bool = True
    
    # Alert thresholds (optional overrides)
    yellow_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    red_threshold: Optional[float] = Field(default=None, ge=0, le=1)


# ============================================================================
# Exit Configuration
# ============================================================================

class ExitConfig(BaseModel):
    """Exit point configuration."""
    id: str
    name: Optional[str] = None
    location: GeoCoord
    capacity: float = Field(default=10.0, gt=0, description="Agents per second capacity")


# ============================================================================
# Venue Configuration
# ============================================================================

class VenueMetadata(BaseModel):
    """Venue metadata."""
    name: str
    description: Optional[str] = None
    location: str = Field(default="Unknown", description="City/area name")
    event_type: Optional[str] = Field(default=None, description="Festival, Concert, etc.")


class SimulationDefaults(BaseModel):
    """Default simulation parameters for this venue."""
    agent_count: Optional[int] = Field(default=None, gt=0, alias="agent_count")
    spawn_rate: Optional[float] = Field(default=None, gt=0)
    agent_speed: Optional[float] = Field(default=None, gt=0)
    max_agents: int = Field(default=500, gt=0)
    base_inflow_rate: float = Field(default=5.0, gt=0)
    speed_multiplier: float = Field(default=1.0, gt=0)


class VenueConfig(BaseModel):
    """Complete venue configuration."""
    
    # Metadata
    venue: VenueMetadata
    
    # Origin point for coordinate conversion
    origin: GeoCoord
    
    # Infrastructure
    roads: List[RoadConfig] = Field(default_factory=list)
    spawn_points: List[SpawnConfig] = Field(default_factory=list)
    choke_points: List[ChokePointConfig] = Field(default_factory=list)
    exits: List[ExitConfig] = Field(default_factory=list)
    
    # Simulation defaults
    defaults: SimulationDefaults = Field(default_factory=SimulationDefaults)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "VenueConfig":
        """Load venue config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "VenueConfig":
        """Load venue config from dictionary."""
        return cls.model_validate(data)
    
    def to_yaml(self, path: Path):
        """Save venue config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_venue_config(config: VenueConfig) -> List[str]:
    """
    Validate venue configuration for logical consistency.
    Returns list of warning/error messages.
    """
    issues = []
    
    road_ids = {r.id for r in config.roads}
    exit_ids = {e.id for e in config.exits}
    
    # Check spawn points reference valid roads/exits
    for spawn in config.spawn_points:
        if spawn.road_id and spawn.road_id not in road_ids:
            issues.append(f"Spawn '{spawn.id}' references unknown road '{spawn.road_id}'")
        if spawn.destination_road_id and spawn.destination_road_id not in road_ids:
            issues.append(f"Spawn '{spawn.id}' destination references unknown road '{spawn.destination_road_id}'")
        if spawn.exit_id and spawn.exit_id not in exit_ids:
            issues.append(f"Spawn '{spawn.id}' references unknown exit '{spawn.exit_id}'")
    
    # Check for at least one spawn point
    if not config.spawn_points:
        issues.append("No spawn points defined")
    
    # Check for at least one choke point
    if not config.choke_points:
        issues.append("No choke points defined - CII cannot be computed")
    
    return issues
