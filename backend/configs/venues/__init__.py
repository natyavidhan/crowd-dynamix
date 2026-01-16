"""Venue configuration package."""

from .schema import (
    VenueConfig,
    VenueMetadata,
    RoadConfig,
    SpawnConfig,
    ChokePointConfig,
    ExitConfig,
    GeoCoord,
    SimulationDefaults,
    validate_venue_config,
)

from .loader import (
    CoordinateSystem,
    Road,
    RoadNetwork,
    ProcessedSpawnPoint,
    VenueLoader,
    list_available_venues,
    load_venue,
)

__all__ = [
    # Schema
    "VenueConfig",
    "VenueMetadata", 
    "RoadConfig",
    "SpawnConfig",
    "ChokePointConfig",
    "ExitConfig",
    "GeoCoord",
    "SimulationDefaults",
    "validate_venue_config",
    # Loader
    "CoordinateSystem",
    "Road",
    "RoadNetwork",
    "ProcessedSpawnPoint",
    "VenueLoader",
    "list_available_venues",
    "load_venue",
]
