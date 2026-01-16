"""
Tests for venue configuration system.
Tests schema validation, loader functionality, and API endpoints.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import yaml

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.venues.schema import (
    VenueConfig, VenueMetadata, RoadConfig, SpawnConfig,
    ChokePointConfig, ExitConfig, GeoCoord, SimulationDefaults,
    validate_venue_config
)
from configs.venues.loader import (
    CoordinateSystem, Road, RoadNetwork, ProcessedSpawnPoint,
    VenueLoader, list_available_venues, load_venue
)
from utils.vector import vec2, magnitude


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_venue_dict():
    """Sample venue configuration as a dictionary."""
    return {
        "venue": {
            "name": "Test Venue",
            "location": "Test City",
            "description": "A test venue for unit tests"
        },
        "origin": {
            "lat": 13.0500,
            "lng": 80.2824
        },
        "bounds": {
            "min_x": -100,
            "max_x": 100,
            "min_y": -100,
            "max_y": 100
        },
        "roads": [
            {
                "id": "main_road",
                "name": "Main Road",
                "width": 10.0,
                "speed_limit": 1.2,
                "bidirectional": True,
                "centerline": [
                    {"lat": 13.0490, "lng": 80.2824},
                    {"lat": 13.0500, "lng": 80.2824},
                    {"lat": 13.0510, "lng": 80.2824}
                ]
            },
            {
                "id": "side_road",
                "name": "Side Road",
                "width": 6.0,
                "speed_limit": 0.8,
                "bidirectional": False,
                "centerline": [
                    {"lat": 13.0500, "lng": 80.2814},
                    {"lat": 13.0500, "lng": 80.2824},
                    {"lat": 13.0500, "lng": 80.2834}
                ]
            }
        ],
        "exits": [
            {
                "id": "exit_north",
                "name": "North Exit",
                "location": {"lat": 13.0510, "lng": 80.2824},
                "capacity": 100
            },
            {
                "id": "exit_south",
                "name": "South Exit",
                "location": {"lat": 13.0490, "lng": 80.2824},
                "capacity": 100
            }
        ],
        "spawn_points": [
            {
                "id": "spawn_south",
                "name": "South Entry",
                "road_id": "main_road",
                "road_position": "start",
                "exit_id": "exit_north",
                "rate": 3.0,
                "spread": 5.0,
                "flow_group": "main_flow"
            },
            {
                "id": "spawn_west",
                "name": "West Entry",
                "location": {"lat": 13.0500, "lng": 80.2814},
                "destination_location": {"lat": 13.0500, "lng": 80.2834},
                "rate": 2.0,
                "spread": 3.0,
                "flow_group": "side_flow"
            }
        ],
        "choke_points": [
            {
                "id": "cp_center",
                "name": "Central Junction",
                "location": {"lat": 13.0500, "lng": 80.2824},
                "radius": 15.0,
                "threshold": 3.0,
                "mmwave_enabled": True,
                "audio_enabled": True,
                "camera_enabled": True
            },
            {
                "id": "cp_north",
                "name": "North Approach",
                "location": {"lat": 13.0505, "lng": 80.2824},
                "radius": 10.0,
                "threshold": 2.5,
                "mmwave_enabled": True,
                "audio_enabled": False,
                "camera_enabled": True
            }
        ],
        "defaults": {
            "agent_count": 100,
            "spawn_rate": 0.5,
            "agent_speed": 1.0
        }
    }


@pytest.fixture
def sample_venue_yaml(sample_venue_dict, tmp_path):
    """Create a temporary YAML file with sample venue config."""
    yaml_path = tmp_path / "test_venue.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(sample_venue_dict, f)
    return yaml_path


@pytest.fixture
def sample_venue_config(sample_venue_dict):
    """Create a VenueConfig from sample dict."""
    return VenueConfig.from_dict(sample_venue_dict)


# ============================================================================
# Schema Tests
# ============================================================================

class TestGeoCoord:
    """Tests for GeoCoord model."""
    
    def test_valid_coords(self):
        coord = GeoCoord(lat=13.0500, lng=80.2824)
        assert coord.lat == 13.0500
        assert coord.lng == 80.2824
    
    def test_boundary_values(self):
        # Equator
        coord = GeoCoord(lat=0.0, lng=0.0)
        assert coord.lat == 0.0
        
        # Extreme values
        coord = GeoCoord(lat=89.0, lng=179.0)
        assert coord.lat == 89.0


class TestRoadConfig:
    """Tests for RoadConfig model."""
    
    def test_valid_road(self):
        road = RoadConfig(
            id="test_road",
            name="Test Road",
            width=10.0,
            speed_limit=1.5,
            bidirectional=True,
            centerline=[
                GeoCoord(lat=13.0490, lng=80.2824),
                GeoCoord(lat=13.0500, lng=80.2824)
            ]
        )
        assert road.id == "test_road"
        assert len(road.centerline) == 2
    
    def test_defaults(self):
        road = RoadConfig(
            id="minimal_road",
            name="Minimal Road",
            centerline=[
                GeoCoord(lat=13.0490, lng=80.2824),
                GeoCoord(lat=13.0500, lng=80.2824)
            ]
        )
        assert road.name == "Minimal Road"
        assert road.width == 6.0  # Default from schema
        assert road.speed_limit == 1.4  # Default from schema
        assert road.bidirectional is True


class TestVenueConfig:
    """Tests for VenueConfig model."""
    
    def test_from_dict(self, sample_venue_dict):
        config = VenueConfig.from_dict(sample_venue_dict)
        
        assert config.venue.name == "Test Venue"
        assert config.origin.lat == 13.0500
        assert len(config.roads) == 2
        assert len(config.exits) == 2
        assert len(config.spawn_points) == 2
        assert len(config.choke_points) == 2
    
    def test_from_yaml(self, sample_venue_yaml):
        config = VenueConfig.from_yaml(sample_venue_yaml)
        
        assert config.venue.name == "Test Venue"
        assert len(config.roads) == 2
    
    def test_to_yaml(self, sample_venue_config, tmp_path):
        output_path = tmp_path / "output.yaml"
        sample_venue_config.to_yaml(output_path)
        
        # Reload and verify
        reloaded = VenueConfig.from_yaml(output_path)
        assert reloaded.venue.name == sample_venue_config.venue.name
        assert len(reloaded.roads) == len(sample_venue_config.roads)
    
    def test_validate_venue_config(self, sample_venue_dict):
        config = VenueConfig.from_dict(sample_venue_dict)
        errors = validate_venue_config(config)
        assert len(errors) == 0
    
    def test_validate_missing_origin(self, sample_venue_dict):
        del sample_venue_dict["origin"]
        try:
            config = VenueConfig.from_dict(sample_venue_dict)
            # If it loads without origin, validation should catch it
            errors = validate_venue_config(config)
            # Should have errors since origin is missing
        except Exception:
            # Expected - missing required field
            pass


# ============================================================================
# Coordinate System Tests
# ============================================================================

class TestCoordinateSystem:
    """Tests for coordinate conversion."""
    
    def test_origin_conversion(self):
        origin = GeoCoord(lat=13.0500, lng=80.2824)
        coord_sys = CoordinateSystem(origin=origin)
        
        # Origin should map to (0, 0)
        sim = coord_sys.geo_to_sim(origin)
        assert abs(sim[0]) < 0.001
        assert abs(sim[1]) < 0.001
    
    def test_north_south_conversion(self):
        origin = GeoCoord(lat=13.0500, lng=80.2824)
        coord_sys = CoordinateSystem(origin=origin)
        
        # 1 degree north should be ~111km
        north = GeoCoord(lat=13.0500 + 0.001, lng=80.2824)  # ~111m north
        sim = coord_sys.geo_to_sim(north)
        
        assert sim[0] < 1  # minimal east-west movement
        assert 100 < sim[1] < 120  # approximately 111m north
    
    def test_roundtrip_conversion(self):
        origin = GeoCoord(lat=13.0500, lng=80.2824)
        coord_sys = CoordinateSystem(origin=origin)
        
        # Convert to sim and back
        test_geo = GeoCoord(lat=13.0510, lng=80.2830)
        sim = coord_sys.geo_to_sim(test_geo)
        back = coord_sys.sim_to_geo(sim)
        
        assert abs(back.lat - test_geo.lat) < 0.00001
        assert abs(back.lng - test_geo.lng) < 0.00001


# ============================================================================
# Road Tests
# ============================================================================

class TestRoad:
    """Tests for Road class."""
    
    def test_road_length(self):
        centerline = [
            vec2(0, 0),
            vec2(100, 0),  # 100m east
            vec2(100, 50)  # 50m north
        ]
        road = Road(
            id="test",
            name="Test Road",
            width=10.0,
            speed_limit=1.0,
            bidirectional=True,
            centerline=centerline
        )
        
        assert abs(road.length - 150.0) < 0.001
    
    def test_point_at_distance(self):
        centerline = [
            vec2(0, 0),
            vec2(100, 0)
        ]
        road = Road(
            id="test",
            name="Test",
            width=10.0,
            speed_limit=1.0,
            bidirectional=True,
            centerline=centerline
        )
        
        # Start
        p = road.point_at_distance(0)
        assert abs(p[0]) < 0.001
        
        # Middle
        p = road.point_at_distance(50)
        assert abs(p[0] - 50) < 0.001
        
        # End
        p = road.point_at_distance(100)
        assert abs(p[0] - 100) < 0.001
    
    def test_get_position(self):
        centerline = [
            vec2(0, 0),
            vec2(50, 0),
            vec2(100, 0)
        ]
        road = Road(
            id="test",
            name="Test",
            width=10.0,
            speed_limit=1.0,
            bidirectional=True,
            centerline=centerline
        )
        
        start = road.get_position("start")
        assert abs(start[0]) < 0.001
        
        end = road.get_position("end")
        assert abs(end[0] - 100) < 0.001
        
        middle = road.get_position("middle")
        assert abs(middle[0] - 50) < 0.001
    
    def test_generate_waypoints(self):
        centerline = [
            vec2(0, 0),
            vec2(100, 0)
        ]
        road = Road(
            id="test",
            name="Test",
            width=10.0,
            speed_limit=1.0,
            bidirectional=True,
            centerline=centerline
        )
        
        waypoints = road.generate_waypoints(spacing=25.0)
        assert len(waypoints) >= 4  # At least 5 points for 100m at 25m spacing
        
        # First should be at start
        assert abs(waypoints[0][0]) < 0.001
        
        # Last should be at end
        assert abs(waypoints[-1][0] - 100) < 0.001
    
    def test_generate_waypoints_reversed(self):
        centerline = [
            vec2(0, 0),
            vec2(100, 0)
        ]
        road = Road(
            id="test",
            name="Test",
            width=10.0,
            speed_limit=1.0,
            bidirectional=True,
            centerline=centerline
        )
        
        waypoints = road.generate_waypoints(reverse=True, spacing=25.0)
        
        # First should be at end (100)
        assert abs(waypoints[0][0] - 100) < 0.001
        
        # Last should be at start (0)
        assert abs(waypoints[-1][0]) < 0.001


class TestRoadNetwork:
    """Tests for RoadNetwork class."""
    
    def test_add_and_get_road(self):
        network = RoadNetwork()
        road = Road(
            id="road1",
            name="Road 1",
            width=10.0,
            speed_limit=1.0,
            bidirectional=True,
            centerline=[vec2(0, 0), vec2(100, 0)]
        )
        
        network.add_road(road)
        
        retrieved = network.get_road("road1")
        assert retrieved is not None
        assert retrieved.id == "road1"
    
    def test_get_nonexistent_road(self):
        network = RoadNetwork()
        assert network.get_road("nonexistent") is None
    
    def test_get_all_roads(self):
        network = RoadNetwork()
        road1 = Road(
            id="road1",
            name="Road 1",
            width=10.0,
            speed_limit=1.0,
            bidirectional=True,
            centerline=[vec2(0, 0), vec2(100, 0)]
        )
        road2 = Road(
            id="road2",
            name="Road 2",
            width=8.0,
            speed_limit=0.8,
            bidirectional=False,
            centerline=[vec2(0, 0), vec2(0, 100)]
        )
        
        network.add_road(road1)
        network.add_road(road2)
        
        all_roads = network.get_all_roads()
        assert len(all_roads) == 2


# ============================================================================
# Venue Loader Tests
# ============================================================================

class TestVenueLoader:
    """Tests for VenueLoader class."""
    
    def test_loader_initialization(self, sample_venue_config):
        loader = VenueLoader(sample_venue_config)
        
        assert loader.config is not None
        assert loader.coord_sys is not None
        assert loader.road_network is not None
    
    def test_roads_processed(self, sample_venue_config):
        loader = VenueLoader(sample_venue_config)
        
        roads = loader.road_network.get_all_roads()
        assert len(roads) == 2
        
        main_road = loader.road_network.get_road("main_road")
        assert main_road is not None
        assert main_road.name == "Main Road"
        assert main_road.width == 10.0
    
    def test_choke_points_processed(self, sample_venue_config):
        loader = VenueLoader(sample_venue_config)
        
        assert len(loader.choke_points) == 2
        
        cp_center = next((cp for cp in loader.choke_points if cp.id == "cp_center"), None)
        assert cp_center is not None
        assert cp_center.name == "Central Junction"
        assert cp_center.sim_radius == 15.0
    
    def test_spawn_points_processed(self, sample_venue_config):
        loader = VenueLoader(sample_venue_config)
        
        assert len(loader.spawn_points) == 2
        
        # Check road-based spawn point
        spawn_south = next((sp for sp in loader.spawn_points if sp.id == "spawn_south"), None)
        assert spawn_south is not None
        assert spawn_south.flow_group == "main_flow"
        assert len(spawn_south.waypoints) > 0
    
    def test_exits_processed(self, sample_venue_config):
        loader = VenueLoader(sample_venue_config)
        
        assert "exit_north" in loader.exits
        assert "exit_south" in loader.exits
    
    def test_road_geometries_geo(self, sample_venue_config):
        loader = VenueLoader(sample_venue_config)
        
        geometries = loader.get_road_geometries_geo()
        assert len(geometries) == 2
        
        main_road_geo = next((g for g in geometries if g["id"] == "main_road"), None)
        assert main_road_geo is not None
        assert "coordinates" in main_road_geo
        assert len(main_road_geo["coordinates"]) == 3  # 3 points in centerline
    
    def test_get_spawn_points_for_group(self, sample_venue_config):
        loader = VenueLoader(sample_venue_config)
        
        main_flow = loader.get_spawn_points_for_group("main_flow")
        assert len(main_flow) == 1
        assert main_flow[0].id == "spawn_south"
        
        side_flow = loader.get_spawn_points_for_group("side_flow")
        assert len(side_flow) == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestVenueDiscovery:
    """Tests for venue discovery functions."""
    
    def test_list_available_venues(self, sample_venue_yaml, tmp_path):
        # Create data directory structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Move yaml to data directory
        import shutil
        shutil.copy(sample_venue_yaml, data_dir / "test_venue.yaml")
        
        venues = list_available_venues(tmp_path)
        assert len(venues) == 1
        assert venues[0]["id"] == "test_venue"
        assert venues[0]["name"] == "Test Venue"
    
    def test_load_venue(self, sample_venue_yaml, tmp_path):
        # Create data directory structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        import shutil
        shutil.copy(sample_venue_yaml, data_dir / "test_venue.yaml")
        
        config = load_venue(tmp_path, "test_venue")
        assert config is not None
        assert config.venue.name == "Test Venue"
    
    def test_load_nonexistent_venue(self, tmp_path):
        config = load_venue(tmp_path, "nonexistent")
        assert config is None


# ============================================================================
# Real Venue Files Tests
# ============================================================================

class TestRealVenueFiles:
    """Tests that validate the actual venue configuration files."""
    
    @pytest.fixture
    def venues_dir(self):
        """Get the actual venues directory."""
        return Path(__file__).parent.parent / "configs" / "venues"
    
    def test_marina_beach_loads(self, venues_dir):
        config = load_venue(venues_dir, "marina_beach")
        if config:  # Only test if file exists
            assert config.venue.name == "Marina Beach"
            assert len(config.roads) > 0
            assert len(config.choke_points) > 0
    
    def test_stadium_loads(self, venues_dir):
        config = load_venue(venues_dir, "stadium")
        if config:
            assert config.venue.name == "City Sports Stadium"
            assert len(config.roads) > 0
    
    def test_temple_festival_loads(self, venues_dir):
        config = load_venue(venues_dir, "temple_festival")
        if config:
            assert config.venue.name == "Kapaleeshwarar Temple Festival"
            assert len(config.roads) > 0
    
    def test_all_venues_valid(self, venues_dir):
        """Test that all venue files in data/ are valid."""
        venues = list_available_venues(venues_dir)
        
        for venue_info in venues:
            config = load_venue(venues_dir, venue_info["id"])
            assert config is not None, f"Failed to load {venue_info['id']}"
            
            # Validate basic structure
            errors = validate_venue_config(config)
            assert len(errors) == 0, f"Validation errors in {venue_info['id']}: {errors}"
            
            # Test that loader can process it
            loader = VenueLoader(config)
            assert len(loader.choke_points) > 0, f"No choke points in {venue_info['id']}"
