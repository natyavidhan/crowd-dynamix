"""Simulation modules."""
from .engine import SimulationEngine, CoordinateSystem
from .agents import AgentPool, SpatialHash, SpawnPoint
from .sensors import SensorBuffers, simulate_all_sensors
from .cii import compute_cii, explain_cii, update_risk_state
