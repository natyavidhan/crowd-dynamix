"""
Venue loader and road network management.
Converts venue configs into simulation-ready data structures.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.venues.schema import (
    VenueConfig, RoadConfig, SpawnConfig, ChokePointConfig, ExitConfig, GeoCoord
)
from utils.vector import Vec2, vec2, normalize, magnitude
from models.types import (
    GeoPoint, SimPoint, ChokePoint, CircleGeometry,
    SensorConfig, SensorType, RiskState
)


# ============================================================================
# Coordinate Conversion
# ============================================================================

@dataclass
class CoordinateSystem:
    """Handles geo <-> simulation coordinate conversion."""
    
    origin: GeoCoord
    meters_per_deg_lat: float = 111320.0
    meters_per_deg_lng: float = field(init=False)
    
    def __post_init__(self):
        lat_rad = np.radians(self.origin.lat)
        self.meters_per_deg_lng = 111320.0 * np.cos(lat_rad)
    
    def geo_to_sim(self, geo: GeoCoord) -> Vec2:
        """Convert geographic to simulation coordinates (meters from origin)."""
        x = (geo.lng - self.origin.lng) * self.meters_per_deg_lng
        y = (geo.lat - self.origin.lat) * self.meters_per_deg_lat
        return vec2(x, y)
    
    def sim_to_geo(self, sim: Vec2) -> GeoCoord:
        """Convert simulation to geographic coordinates."""
        lng = self.origin.lng + sim[0] / self.meters_per_deg_lng
        lat = self.origin.lat + sim[1] / self.meters_per_deg_lat
        return GeoCoord(lat=lat, lng=lng)


# ============================================================================
# Road Network
# ============================================================================

@dataclass
class Road:
    """Processed road with simulation coordinates."""
    id: str
    name: str
    width: float
    speed_limit: float
    bidirectional: bool
    
    # Centerline in simulation coordinates
    centerline: List[Vec2]
    
    # Precomputed data
    length: float = field(init=False)
    segments: List[Tuple[Vec2, Vec2, float]] = field(init=False)  # (start, end, length)
    
    def __post_init__(self):
        """Compute road segments and total length."""
        self.segments = []
        self.length = 0.0
        
        for i in range(len(self.centerline) - 1):
            start = self.centerline[i]
            end = self.centerline[i + 1]
            seg_length = magnitude(end - start)
            self.segments.append((start, end, seg_length))
            self.length += seg_length
    
    def point_at_distance(self, distance: float) -> Vec2:
        """Get point on road at given distance from start."""
        distance = max(0, min(distance, self.length))
        
        cumulative = 0.0
        for start, end, seg_length in self.segments:
            if cumulative + seg_length >= distance:
                t = (distance - cumulative) / seg_length if seg_length > 0 else 0
                return start + (end - start) * t
            cumulative += seg_length
        
        return self.centerline[-1]
    
    def direction_at_distance(self, distance: float) -> Vec2:
        """Get direction vector at given distance from start."""
        cumulative = 0.0
        for start, end, seg_length in self.segments:
            if cumulative + seg_length >= distance:
                return normalize(end - start)
            cumulative += seg_length
        
        if len(self.segments) > 0:
            start, end, _ = self.segments[-1]
            return normalize(end - start)
        return vec2(1, 0)
    
    def get_position(self, position: str) -> Vec2:
        """Get position by name: 'start', 'end', 'middle'."""
        if position == "start":
            return self.centerline[0].copy()
        elif position == "end":
            return self.centerline[-1].copy()
        elif position == "middle":
            return self.point_at_distance(self.length / 2)
        else:
            return self.centerline[0].copy()
    
    def generate_waypoints(self, reverse: bool = False, spacing: float = 10.0) -> List[Vec2]:
        """Generate waypoints along the road."""
        waypoints = []
        
        # Sample points along the road
        num_points = max(2, int(self.length / spacing) + 1)
        for i in range(num_points):
            dist = (i / (num_points - 1)) * self.length
            waypoints.append(self.point_at_distance(dist))
        
        if reverse:
            waypoints = waypoints[::-1]
        
        return waypoints


@dataclass
class RoadNetwork:
    """Collection of roads with lookup capabilities."""
    roads: Dict[str, Road] = field(default_factory=dict)
    
    def add_road(self, road: Road):
        self.roads[road.id] = road
    
    def get_road(self, road_id: str) -> Optional[Road]:
        return self.roads.get(road_id)
    
    def get_all_roads(self) -> List[Road]:
        return list(self.roads.values())


# ============================================================================
# Spawn Point Processing
# ============================================================================

@dataclass
class ProcessedSpawnPoint:
    """Spawn point ready for simulation."""
    id: str
    name: str
    position: Vec2
    direction: float  # radians
    waypoints: List[Vec2]
    rate: float
    spread: float
    flow_group: str


# ============================================================================
# Venue Loader
# ============================================================================

class VenueLoader:
    """Loads and processes venue configurations."""
    
    def __init__(self, config: VenueConfig):
        self.config = config
        self.coord_sys = CoordinateSystem(origin=config.origin)
        self.road_network = RoadNetwork()
        self.spawn_points: List[ProcessedSpawnPoint] = []
        self.choke_points: List[ChokePoint] = []
        self.exits: Dict[str, Vec2] = {}
        
        self._process_roads()
        self._process_exits()
        self._process_spawn_points()
        self._process_choke_points()
    
    def _process_roads(self):
        """Convert road configs to Road objects."""
        for road_config in self.config.roads:
            centerline = [
                self.coord_sys.geo_to_sim(coord)
                for coord in road_config.centerline
            ]
            
            road = Road(
                id=road_config.id,
                name=road_config.name,
                width=road_config.width,
                speed_limit=road_config.speed_limit,
                bidirectional=road_config.bidirectional,
                centerline=centerline
            )
            
            self.road_network.add_road(road)
    
    def _process_exits(self):
        """Convert exit configs to positions."""
        for exit_config in self.config.exits:
            pos = self.coord_sys.geo_to_sim(exit_config.location)
            self.exits[exit_config.id] = pos
    
    def _process_spawn_points(self):
        """Convert spawn configs to ProcessedSpawnPoints."""
        for spawn_config in self.config.spawn_points:
            # Determine spawn position
            if spawn_config.road_id:
                road = self.road_network.get_road(spawn_config.road_id)
                if road:
                    position = road.get_position(spawn_config.road_position or "start")
                    
                    # Generate waypoints along road
                    reverse = spawn_config.road_position == "end"
                    waypoints = road.generate_waypoints(reverse=reverse)
                    
                    # Add exit as final waypoint if specified
                    if spawn_config.exit_id and spawn_config.exit_id in self.exits:
                        waypoints.append(self.exits[spawn_config.exit_id])
                    
                    # Calculate direction
                    if len(waypoints) >= 2:
                        direction = np.arctan2(
                            waypoints[1][1] - waypoints[0][1],
                            waypoints[1][0] - waypoints[0][0]
                        )
                    else:
                        direction = 0.0
                else:
                    continue  # Skip invalid road reference
            elif spawn_config.location:
                position = self.coord_sys.geo_to_sim(spawn_config.location)
                
                # Create waypoints to destination
                waypoints = [position]
                if spawn_config.destination_location:
                    dest = self.coord_sys.geo_to_sim(spawn_config.destination_location)
                    waypoints.append(dest)
                elif spawn_config.exit_id and spawn_config.exit_id in self.exits:
                    waypoints.append(self.exits[spawn_config.exit_id])
                
                direction = 0.0
                if len(waypoints) >= 2:
                    direction = np.arctan2(
                        waypoints[1][1] - waypoints[0][1],
                        waypoints[1][0] - waypoints[0][0]
                    )
            else:
                continue  # No valid position
            
            processed = ProcessedSpawnPoint(
                id=spawn_config.id,
                name=spawn_config.name or spawn_config.id,
                position=position,
                direction=direction,
                waypoints=waypoints,
                rate=spawn_config.rate,
                spread=spawn_config.spread,
                flow_group=spawn_config.flow_group
            )
            
            self.spawn_points.append(processed)
    
    def _process_choke_points(self):
        """Convert choke point configs to ChokePoint objects."""
        for cp_config in self.config.choke_points:
            sim_pos = self.coord_sys.geo_to_sim(cp_config.location)
            
            sensors = []
            if cp_config.mmwave_enabled:
                sensors.append(SensorConfig(type=SensorType.MMWAVE, enabled=True))
            if cp_config.audio_enabled:
                sensors.append(SensorConfig(type=SensorType.AUDIO, enabled=True))
            if cp_config.camera_enabled:
                sensors.append(SensorConfig(type=SensorType.CAMERA, enabled=True))
            
            choke_point = ChokePoint(
                id=cp_config.id,
                name=cp_config.name,
                center=GeoPoint(lat=cp_config.location.lat, lng=cp_config.location.lng),
                geometry=CircleGeometry(radius=cp_config.radius),
                sensors=sensors,
                sim_center=SimPoint(x=sim_pos[0], y=sim_pos[1]),
                sim_radius=cp_config.radius,
                risk_state=RiskState()
            )
            
            self.choke_points.append(choke_point)
    
    def get_spawn_points_for_group(self, group: str) -> List[ProcessedSpawnPoint]:
        """Get spawn points belonging to a flow group."""
        return [sp for sp in self.spawn_points if sp.flow_group == group]
    
    def get_road_geometries_geo(self) -> List[Dict]:
        """Get road geometries in geographic coordinates for frontend display."""
        geometries = []
        
        for road in self.road_network.get_all_roads():
            # Convert centerline back to geo
            coords = [
                self.coord_sys.sim_to_geo(point)
                for point in road.centerline
            ]
            
            geometries.append({
                "id": road.id,
                "name": road.name,
                "width": road.width,
                "coordinates": [{"lat": c.lat, "lng": c.lng} for c in coords]
            })
        
        return geometries


# ============================================================================
# Config Discovery
# ============================================================================

def list_available_venues(configs_dir: Path) -> List[Dict]:
    """List all available venue configs."""
    venues = []
    
    data_dir = configs_dir / "data"
    if not data_dir.exists():
        return venues
    
    for file_path in data_dir.glob("*.yaml"):
        try:
            config = VenueConfig.from_yaml(file_path)
            venues.append({
                "id": file_path.stem,
                "name": config.venue.name,
                "location": config.venue.location,
                "description": config.venue.description,
                "file": str(file_path)
            })
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    
    return venues


def load_venue(configs_dir: Path, venue_id: str) -> Optional[VenueConfig]:
    """Load a venue config by ID."""
    file_path = configs_dir / "data" / f"{venue_id}.yaml"
    if file_path.exists():
        return VenueConfig.from_yaml(file_path)
    return None
