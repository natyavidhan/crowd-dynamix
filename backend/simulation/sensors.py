"""
Sensor simulation module.
Generates synthetic mmWave, audio, and camera sensor data from agent state.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from collections import deque
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector import (
    Vec2, vec2, magnitude, angle_of, circular_mean, circular_variance
)
from models.types import (
    ChokePoint, MmWaveSensorData, AudioSensorData, CameraSensorData,
    AggregatedSensorData, SensorType, CircleGeometry, PolygonGeometry
)
from simulation.agents import AgentPool, MIN_SPEED


# ============================================================================
# Constants
# ============================================================================

MAX_DENSITY = 6.0  # people/m² (crush conditions)
NOISE_LEVEL = 0.05  # sensor noise amplitude


# ============================================================================
# Geometry Utilities
# ============================================================================

def point_in_circle(
    point: Vec2,
    center: Vec2,
    radius: float
) -> bool:
    """Check if point is inside circle."""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return dx*dx + dy*dy <= radius*radius


def point_in_polygon(point: Vec2, vertices: List[Vec2]) -> bool:
    """Check if point is inside polygon using ray casting."""
    n = len(vertices)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        
        if ((yi > point[1]) != (yj > point[1])) and \
           (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def circle_area(radius: float) -> float:
    """Area of circle."""
    return np.pi * radius * radius


def polygon_area(vertices: List[Vec2]) -> float:
    """Area of polygon using shoelace formula."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


# ============================================================================
# Agent Queries
# ============================================================================

def get_agents_in_choke_point(
    pool: AgentPool,
    choke_point: ChokePoint
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get indices, positions, and velocities of agents in a choke point.
    Returns (indices, positions, velocities).
    """
    if choke_point.sim_center is None:
        return np.array([]), np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    center = vec2(choke_point.sim_center.x, choke_point.sim_center.y)
    active_indices = pool.get_active_indices()
    
    if len(active_indices) == 0:
        return np.array([]), np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    in_region = []
    
    if isinstance(choke_point.geometry, CircleGeometry) or choke_point.geometry.type == "circle":
        radius = choke_point.sim_radius or choke_point.geometry.radius
        for idx in active_indices:
            if point_in_circle(pool.positions[idx], center, radius):
                in_region.append(idx)
    else:
        # Polygon - would need sim_vertices conversion
        # For now, use bounding circle approximation
        radius = choke_point.sim_radius or 10.0
        for idx in active_indices:
            if point_in_circle(pool.positions[idx], center, radius):
                in_region.append(idx)
    
    if not in_region:
        return np.array([]), np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    indices = np.array(in_region, dtype=np.int32)
    positions = pool.positions[indices]
    velocities = pool.velocities[indices]
    
    return indices, positions, velocities


def get_choke_point_area(choke_point: ChokePoint) -> float:
    """Get area of choke point in m²."""
    if isinstance(choke_point.geometry, CircleGeometry) or choke_point.geometry.type == "circle":
        return circle_area(choke_point.geometry.radius)
    else:
        # Would need polygon vertices
        return 100.0  # Default fallback


# ============================================================================
# Time Series Buffers for Stop-Go Detection
# ============================================================================

class VelocityBuffer:
    """Circular buffer for velocity history."""
    
    def __init__(self, max_samples: int = 300):
        self.max_samples = max_samples
        self.timestamps: deque = deque(maxlen=max_samples)
        self.avg_velocities: deque = deque(maxlen=max_samples)
        self.stationary_ratios: deque = deque(maxlen=max_samples)
    
    def push(
        self,
        timestamp: float,
        avg_velocity: float,
        stationary_ratio: float
    ):
        self.timestamps.append(timestamp)
        self.avg_velocities.append(avg_velocity)
        self.stationary_ratios.append(stationary_ratio)
    
    def compute_stop_go_frequency(self, window_seconds: float = 30.0) -> float:
        """
        Count velocity transitions (moving <-> stopped) in window.
        Returns events per minute.
        """
        if len(self.timestamps) < 2:
            return 0.0
        
        now = time.time()
        cutoff = now - window_seconds
        
        # Find samples in window
        transitions = 0
        prev_moving = None
        
        for i, (ts, vel) in enumerate(zip(self.timestamps, self.avg_velocities)):
            if ts < cutoff:
                continue
            
            is_moving = vel > MIN_SPEED
            if prev_moving is not None and is_moving != prev_moving:
                transitions += 1
            prev_moving = is_moving
        
        # Convert to per-minute rate
        return transitions * (60.0 / window_seconds)


class SensorBuffers:
    """Manages velocity buffers for all choke points."""
    
    def __init__(self):
        self.buffers: Dict[str, VelocityBuffer] = {}
    
    def get_buffer(self, choke_point_id: str) -> VelocityBuffer:
        if choke_point_id not in self.buffers:
            self.buffers[choke_point_id] = VelocityBuffer()
        return self.buffers[choke_point_id]


# ============================================================================
# Sensor Simulators
# ============================================================================

def add_noise(value: float, noise_level: float = NOISE_LEVEL) -> float:
    """Add Gaussian noise to a value."""
    return value + np.random.normal(0, noise_level)


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def simulate_mmwave(
    choke_point: ChokePoint,
    pool: AgentPool,
    buffers: SensorBuffers
) -> Optional[MmWaveSensorData]:
    """
    Simulate mmWave radar sensor reading.
    Primary sensor - extracts velocity and motion patterns.
    """
    indices, positions, velocities = get_agents_in_choke_point(pool, choke_point)
    
    timestamp = time.time()
    buffer = buffers.get_buffer(choke_point.id)
    
    if len(indices) == 0:
        # No agents in region
        buffer.push(timestamp, 0.0, 1.0)
        return MmWaveSensorData(
            timestamp=timestamp,
            choke_point_id=choke_point.id,
            avg_velocity=0.0,
            velocity_variance=0.0,
            dominant_direction=0.0,
            directional_divergence=0.0,
            stop_go_frequency=buffer.compute_stop_go_frequency(),
            stationary_ratio=1.0
        )
    
    # Compute velocity statistics
    speeds = np.linalg.norm(velocities, axis=1)
    avg_velocity = float(np.mean(speeds))
    velocity_variance = float(np.var(speeds))
    
    # Direction analysis
    moving_mask = speeds > MIN_SPEED
    if np.any(moving_mask):
        directions = np.arctan2(velocities[moving_mask, 1], velocities[moving_mask, 0])
        dominant_direction = circular_mean(directions)
        directional_divergence = circular_variance(directions)
    else:
        dominant_direction = 0.0
        directional_divergence = 0.0
    
    # Stationary ratio
    stationary_count = np.sum(~moving_mask)
    stationary_ratio = stationary_count / len(indices)
    
    # Update buffer and compute stop-go
    buffer.push(timestamp, avg_velocity, stationary_ratio)
    stop_go_frequency = buffer.compute_stop_go_frequency()
    
    # Add sensor noise
    return MmWaveSensorData(
        timestamp=timestamp,
        choke_point_id=choke_point.id,
        avg_velocity=add_noise(avg_velocity, 0.05),
        velocity_variance=add_noise(velocity_variance, 0.02),
        dominant_direction=dominant_direction,
        directional_divergence=clamp(add_noise(directional_divergence, 0.03)),
        stop_go_frequency=max(0, add_noise(stop_go_frequency, 0.5)),
        stationary_ratio=clamp(add_noise(stationary_ratio, 0.02))
    )


def simulate_audio(
    choke_point: ChokePoint,
    pool: AgentPool
) -> Optional[AudioSensorData]:
    """
    Simulate audio sensor reading.
    Supporting sensor - detects crowd stress through sound.
    """
    indices, positions, velocities = get_agents_in_choke_point(pool, choke_point)
    
    timestamp = time.time()
    
    if len(indices) == 0:
        return AudioSensorData(
            timestamp=timestamp,
            choke_point_id=choke_point.id,
            sound_energy_level=0.1,  # ambient noise
            spike_detected=False,
            spike_intensity=0.0,
            audio_character="ambient"
        )
    
    # Density affects base sound level
    area = get_choke_point_area(choke_point)
    density = len(indices) / area
    base_sound = min(1.0, density / MAX_DENSITY)
    
    # Distress indicator: agents with low patience
    patience_values = pool.patience[indices]
    distressed_ratio = np.mean(patience_values < 0.3)
    
    # Stopped/pushing agents contribute to distress sound
    states = pool.states[indices]
    pushing_ratio = np.mean(states == 3)  # pushing state
    
    # Combined sound level
    sound_level = base_sound + distressed_ratio * 0.2 + pushing_ratio * 0.3
    sound_level = clamp(sound_level)
    
    # Spike detection (stochastic)
    spike_probability = (distressed_ratio + pushing_ratio) * 0.1
    spike_detected = np.random.random() < spike_probability
    spike_intensity = (0.5 + np.random.random() * 0.5) if spike_detected else 0.0
    
    # Character classification
    if sound_level > 0.7 or pushing_ratio > 0.3:
        character = "distressed"
    elif sound_level > 0.4:
        character = "loud"
    else:
        character = "ambient"
    
    return AudioSensorData(
        timestamp=timestamp,
        choke_point_id=choke_point.id,
        sound_energy_level=clamp(add_noise(sound_level, 0.05)),
        spike_detected=spike_detected,
        spike_intensity=spike_intensity,
        audio_character=character
    )


def simulate_camera(
    choke_point: ChokePoint,
    pool: AgentPool
) -> Optional[CameraSensorData]:
    """
    Simulate camera/vision sensor reading.
    Contextual sensor - provides density and flow information.
    """
    indices, positions, velocities = get_agents_in_choke_point(pool, choke_point)
    
    timestamp = time.time()
    area = get_choke_point_area(choke_point)
    
    if len(indices) == 0:
        return CameraSensorData(
            timestamp=timestamp,
            choke_point_id=choke_point.id,
            crowd_density=0.0,
            occlusion_percentage=0.0,
            optical_flow_consistency=1.0
        )
    
    # Density
    crowd_density = len(indices) / area
    
    # Occlusion increases with density (simulates camera view blockage)
    occlusion = min(0.95, crowd_density / MAX_DENSITY)
    
    # Optical flow consistency from velocity alignment
    speeds = np.linalg.norm(velocities, axis=1)
    moving_mask = speeds > MIN_SPEED
    
    if np.any(moving_mask):
        directions = np.arctan2(velocities[moving_mask, 1], velocities[moving_mask, 0])
        divergence = circular_variance(directions)
        optical_flow = 1.0 - divergence
    else:
        optical_flow = 1.0  # No movement = consistent (static)
    
    return CameraSensorData(
        timestamp=timestamp,
        choke_point_id=choke_point.id,
        crowd_density=add_noise(crowd_density, 0.1),
        occlusion_percentage=clamp(add_noise(occlusion, 0.02)),
        optical_flow_consistency=clamp(add_noise(optical_flow, 0.03))
    )


def simulate_all_sensors(
    choke_point: ChokePoint,
    pool: AgentPool,
    buffers: SensorBuffers
) -> AggregatedSensorData:
    """
    Simulate all enabled sensors for a choke point.
    Returns aggregated sensor data.
    """
    mmwave_data = None
    audio_data = None
    camera_data = None
    
    # Check which sensors are enabled
    for sensor_config in choke_point.sensors:
        if not sensor_config.enabled:
            continue
        
        if sensor_config.type == SensorType.MMWAVE:
            mmwave_data = simulate_mmwave(choke_point, pool, buffers)
        elif sensor_config.type == SensorType.AUDIO:
            audio_data = simulate_audio(choke_point, pool)
        elif sensor_config.type == SensorType.CAMERA:
            camera_data = simulate_camera(choke_point, pool)
    
    # If no sensors configured, simulate all
    if not choke_point.sensors:
        mmwave_data = simulate_mmwave(choke_point, pool, buffers)
        audio_data = simulate_audio(choke_point, pool)
        camera_data = simulate_camera(choke_point, pool)
    
    # Compute overall confidence
    confidences = [s.confidence for s in choke_point.sensors if s.enabled]
    overall_confidence = np.mean(confidences) if confidences else 0.95
    
    return AggregatedSensorData(
        choke_point_id=choke_point.id,
        timestamp=time.time(),
        mmwave=mmwave_data,
        audio=audio_data,
        camera=camera_data,
        overall_confidence=overall_confidence
    )
