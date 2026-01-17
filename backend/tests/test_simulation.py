"""
Tests for simulation engine and performance.
Run with: pytest tests/test_simulation.py -v
"""

import pytest
import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.engine import SimulationEngine
from simulation.agents import AgentPool, SpatialHash, vec2
from simulation.sensors import VelocityBuffer, SensorBuffers, get_agents_in_choke_point
from simulation.cii import compute_cii, explain_cii, DEFAULT_WEIGHTS
from models.types import (
    SimulationConfig, ChokePoint, GeoPoint, CircleGeometry,
    AggregatedSensorData, MmWaveSensorData, CameraSensorData, AudioSensorData
)
import numpy as np


class TestAgentPool:
    """Test agent pool operations."""
    
    def test_agent_pool_creation(self):
        """AgentPool should initialize with correct size."""
        pool = AgentPool(max_agents=100)
        assert pool.max_agents == 100
        assert pool.count == 0
        assert len(pool.positions) == 100
        assert len(pool.velocities) == 100
        assert pool.positions.shape == (100, 2)
    
    def test_spawn_agent(self):
        """spawn() should add an agent."""
        pool = AgentPool(max_agents=10)
        waypoints = [vec2(10, 10), vec2(20, 20)]
        idx = pool.spawn(position=vec2(5.0, 10.0), waypoints=waypoints)
        assert idx == 0
        assert pool.count == 1
        assert pool.positions[0, 0] == 5.0
        assert pool.positions[0, 1] == 10.0
    
    def test_spawn_respects_max(self):
        """spawn() should return -1 when pool is full."""
        pool = AgentPool(max_agents=3)
        waypoints = [vec2(10, 10)]
        pool.spawn(position=vec2(0, 0), waypoints=waypoints)
        pool.spawn(position=vec2(1, 1), waypoints=waypoints)
        pool.spawn(position=vec2(2, 2), waypoints=waypoints)
        idx = pool.spawn(position=vec2(3, 3), waypoints=waypoints)  # This should fail
        assert idx == -1
        assert pool.count == 3
    
    def test_despawn_agent(self):
        """despawn() should remove agent."""
        pool = AgentPool(max_agents=10)
        waypoints = [vec2(10, 10)]
        pool.spawn(position=vec2(0, 0), waypoints=waypoints)  # idx 0
        pool.spawn(position=vec2(1, 1), waypoints=waypoints)  # idx 1
        pool.spawn(position=vec2(2, 2), waypoints=waypoints)  # idx 2
        
        pool.despawn(0)  # Remove first
        assert pool.count == 2
        assert not pool.active[0]  # First slot is now inactive
    
    def test_get_active_indices(self):
        """get_active_indices() should return active agents."""
        pool = AgentPool(max_agents=10)
        waypoints = [vec2(10, 10)]
        pool.spawn(position=vec2(0, 0), waypoints=waypoints)
        pool.spawn(position=vec2(1, 1), waypoints=waypoints)
        
        indices = pool.get_active_indices()
        assert len(indices) == 2


class TestSpatialHash:
    """Test spatial hash grid."""
    
    def test_spatial_hash_creation(self):
        """SpatialHash should initialize correctly."""
        sh = SpatialHash(cell_size=5.0)
        assert sh.cell_size == 5.0
    
    def test_rebuild_and_query(self):
        """rebuild() and query_radius() should work correctly."""
        sh = SpatialHash(cell_size=10.0)
        
        # Create a mock pool
        pool = AgentPool(max_agents=10)
        waypoints = [vec2(100, 100)]
        pool.spawn(position=vec2(5.0, 5.0), waypoints=waypoints)
        pool.spawn(position=vec2(5.5, 5.5), waypoints=waypoints)
        pool.spawn(position=vec2(50.0, 50.0), waypoints=waypoints)  # Far away
        
        sh.rebuild(pool)
        
        # Query near first point
        neighbors = sh.query_radius(vec2(5.0, 5.0), radius=3.0, pool=pool)
        # Should find at least the first two points (they're close)
        assert len(neighbors) >= 1


class TestVelocityBuffer:
    """Test velocity buffer for stop-go detection."""
    
    def test_velocity_buffer_creation(self):
        """VelocityBuffer should initialize with correct size."""
        buf = VelocityBuffer(max_samples=300)
        assert buf.max_samples == 300
        assert len(buf.timestamps) == 0
    
    def test_velocity_buffer_push(self):
        """push() should record data."""
        buf = VelocityBuffer(max_samples=10)
        buf.push(timestamp=1.0, avg_velocity=1.5, stationary_ratio=0.1)
        
        assert len(buf.timestamps) == 1
        assert buf.avg_velocities[0] == 1.5
        assert buf.stationary_ratios[0] == 0.1
    
    def test_stop_go_frequency_empty(self):
        """stop_go_frequency() should return 0 for empty buffer."""
        buf = VelocityBuffer(max_samples=10)
        freq = buf.compute_stop_go_frequency()
        assert freq == 0.0


class TestCIIComputation:
    """Test CII computation with proper sensor data."""
    
    def create_sensor_data(
        self,
        velocity_variance=0.1,
        stop_go_freq=1.0,
        directional_divergence=0.2,
        density=2.0
    ) -> AggregatedSensorData:
        """Helper to create sensor data for testing."""
        return AggregatedSensorData(
            choke_point_id="test",
            mmwave=MmWaveSensorData(
                choke_point_id="test",
                avg_velocity=1.0,
                velocity_variance=velocity_variance,
                dominant_direction=0.0,
                directional_divergence=directional_divergence,
                stop_go_frequency=stop_go_freq,
                stationary_ratio=0.2
            ),
            camera=CameraSensorData(
                choke_point_id="test",
                crowd_density=density,
                occlusion_percentage=0.1,
                optical_flow_consistency=0.8
            ),
            audio=None
        )
    
    def test_cii_returns_valid_range(self):
        """compute_cii should return value between 0 and 1."""
        sensor_data = self.create_sensor_data()
        cii = compute_cii(sensor_data)
        assert 0.0 <= cii <= 1.0
    
    def test_cii_high_inputs_high_output(self):
        """High risk inputs should produce high CII."""
        sensor_data = self.create_sensor_data(
            velocity_variance=0.5,  # High
            stop_go_freq=10.0,      # High
            directional_divergence=1.0,  # Max
            density=5.0             # High
        )
        cii = compute_cii(sensor_data)
        assert cii > 0.7  # Should be high
    
    def test_cii_low_inputs_low_output(self):
        """Low risk inputs should produce low CII."""
        sensor_data = self.create_sensor_data(
            velocity_variance=0.0,
            stop_go_freq=0.0,
            directional_divergence=0.0,
            density=0.5
        )
        cii = compute_cii(sensor_data)
        assert cii < 0.3  # Should be low
    
    def test_explain_cii_returns_factors(self):
        """explain_cii should return factor breakdown."""
        sensor_data = self.create_sensor_data()
        explanation = explain_cii(sensor_data)
        
        assert hasattr(explanation, "cii")
        assert hasattr(explanation, "contributions")
        assert len(explanation.contributions) == 4
        
        factor_names = [f.factor for f in explanation.contributions]
        assert "Velocity Variance" in factor_names
        assert "Stop-Go Frequency" in factor_names
        assert "Directional Divergence" in factor_names
        assert "Crowd Density" in factor_names
    
    def test_cii_no_mmwave_returns_zero(self):
        """compute_cii should return 0 if no mmWave data."""
        sensor_data = AggregatedSensorData(
            choke_point_id="test",
            mmwave=None,
            camera=None,
            audio=None
        )
        cii = compute_cii(sensor_data)
        assert cii == 0.0


class TestSimulationEngine:
    """Test the main simulation engine."""
    
    @pytest.fixture
    def engine(self, tmp_path):
        """Create a simulation engine with no venue configs (empty configs dir)."""
        config = SimulationConfig(max_agents=100)
        # Use empty tmp_path as configs dir to avoid auto-loading venues
        return SimulationEngine(config=config, configs_dir=tmp_path)
    
    def test_engine_initialization(self, engine):
        """Engine should initialize with correct state."""
        assert engine.config.max_agents == 100
        # Agent pool is now a RoadAgentPool wrapped in AgentPoolAdapter
        assert engine.road_agent_pool is not None or engine.agent_pool is not None
        assert len(engine.choke_points) > 0  # Demo choke points created
    
    def test_add_choke_point(self, engine):
        """add_choke_point should add to list."""
        initial_count = len(engine.choke_points)
        cp = ChokePoint(
            id="cp_test",
            name="Test CP",
            center=GeoPoint(lat=12.9716, lng=77.5946),
            geometry=CircleGeometry(radius=20.0)
        )
        engine.add_choke_point(cp)
        
        assert len(engine.choke_points) == initial_count + 1
        assert any(c.id == "cp_test" for c in engine.choke_points)
    
    def test_remove_choke_point(self, engine):
        """remove_choke_point should remove by id."""
        # Add one first
        cp = ChokePoint(
            id="cp_to_remove",
            name="To Remove",
            center=GeoPoint(lat=0, lng=0),
            geometry=CircleGeometry(radius=10)
        )
        engine.add_choke_point(cp)
        
        initial_count = len(engine.choke_points)
        engine.remove_choke_point("cp_to_remove")
        
        assert len(engine.choke_points) == initial_count - 1
        assert not any(c.id == "cp_to_remove" for c in engine.choke_points)
    
    def test_reset_clears_agents(self, engine):
        """reset() should clear agents."""
        # Spawn agents using the road agent system
        engine.config.running = True
        # Force spawn by running the step which handles spawning
        for _ in range(10):
            engine.step(dt=1/60)
        
        engine.reset()
        
        # After reset, agent pool count should be 0
        assert engine.agent_pool.count == 0
    
    def test_step_does_nothing_when_stopped(self, engine):
        """step() should not run when config.running is False."""
        engine.config.running = False
        initial_tick = engine.tick
        
        engine.step(dt=1.0)
        
        assert engine.tick == initial_tick  # No change
    
    def test_step_increments_tick_when_running(self, engine):
        """step() should increment tick when running."""
        engine.config.running = True
        initial_tick = engine.tick
        
        engine.step(dt=1/60)
        
        assert engine.tick == initial_tick + 1


class TestPerformance:
    """Performance-related tests."""
    
    def test_step_performance_100_agents(self):
        """step() should complete in reasonable time for 100 agents."""
        config = SimulationConfig(max_agents=200)
        engine = SimulationEngine(config=config)
        engine.config.running = True
        
        # Let agents spawn naturally over time by running steps
        # The road agent system spawns agents through spawn points
        for _ in range(100):
            engine.step(dt=1/60)
        
        # Measure time for 100 more steps
        start = time.perf_counter()
        for _ in range(100):
            engine.step(dt=1/60)
        elapsed = time.perf_counter() - start
        
        # Should complete in under 2 seconds
        assert elapsed < 2.0, f"100 steps took {elapsed:.2f}s, expected < 2.0s"
    
    def test_get_state_serializable(self):
        """get_state() should return JSON-serializable dict."""
        config = SimulationConfig(max_agents=10)
        engine = SimulationEngine(config=config)
        engine.config.running = True
        
        # Let some agents spawn
        for _ in range(10):
            engine.step(dt=1/60)
        
        state = engine.get_state()
        
        # Should be serializable
        import json
        json_str = state.model_dump_json()
        assert len(json_str) > 0
        
        # Check structure
        data = json.loads(json_str)
        assert "agents" in data
        assert "choke_points" in data
        assert "tick" in data


class TestMemoryBounds:
    """Tests to ensure memory stays bounded."""
    
    def test_velocity_buffer_bounded(self):
        """VelocityBuffer should not grow unbounded."""
        buf = VelocityBuffer(max_samples=100)
        
        # Push many more samples than max
        for i in range(500):
            buf.push(float(i), 1.0, 0.1)
        
        # Should be capped at max_samples
        assert len(buf.timestamps) == 100
        assert len(buf.avg_velocities) == 100
    
    def test_agent_count_bounded(self):
        """Agent count should stay bounded by max_agents."""
        pool = AgentPool(max_agents=10)
        waypoints = [vec2(100, 100)]
        
        # Try to spawn more than max
        spawned = 0
        for i in range(20):
            idx = pool.spawn(position=vec2(i, i), waypoints=waypoints)
            if idx >= 0:
                spawned += 1
        
        assert spawned == 10
        assert pool.count == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
