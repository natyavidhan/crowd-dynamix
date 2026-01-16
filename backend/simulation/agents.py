"""
Agent simulation with steering behaviors.
Implements crowd physics: path following, separation, collision avoidance, alignment.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector import (
    Vec2, vec2, zero, magnitude, normalize, clamp_magnitude,
    distance, distance_squared, dot, angle_of, from_angle,
    random_in_circle, batch_distances
)


# ============================================================================
# Constants
# ============================================================================

# Movement limits
MAX_SPEED = 1.4          # m/s (normal walking)
MIN_SPEED = 0.1          # m/s (below this = stopped)
MAX_ACCELERATION = 2.0   # m/s²
FRICTION_DECAY = 0.5     # velocity decay per second (half-life style)

# Personal space
AGENT_RADIUS = 0.3       # meters
PERSONAL_SPACE = 0.5     # meters (comfortable distance)
NEIGHBOR_RADIUS = 3.0    # meters (perception range)

# Steering weights
PATH_WEIGHT = 1.0
AVOIDANCE_WEIGHT = 2.5
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 0.3

# Path following
WAYPOINT_ARRIVAL_DIST = 1.0  # meters
SLOWDOWN_DIST = 3.0          # meters (start slowing near waypoint)

# Collision prediction
COLLISION_HORIZON = 2.0  # seconds


# ============================================================================
# Agent Data Structure (optimized for NumPy)
# ============================================================================

@dataclass
class AgentPool:
    """
    Pool of agents stored as NumPy arrays for efficient batch operations.
    Using Structure of Arrays (SoA) layout for performance.
    """
    max_agents: int
    count: int = 0
    
    # Core state (Nx2 arrays)
    positions: np.ndarray = field(init=False)
    velocities: np.ndarray = field(init=False)
    target_velocities: np.ndarray = field(init=False)
    
    # Scalars (N arrays)
    ids: np.ndarray = field(init=False)
    patience: np.ndarray = field(init=False)
    max_speeds: np.ndarray = field(init=False)
    radii: np.ndarray = field(init=False)
    states: np.ndarray = field(init=False)  # 0=moving, 1=slowing, 2=stopped, 3=pushing
    
    # Path following (lists because variable length)
    waypoints: List[List[Vec2]] = field(init=False)
    waypoint_indices: np.ndarray = field(init=False)
    
    # Active mask
    active: np.ndarray = field(init=False)
    
    # ID counter
    _next_id: int = field(default=0, init=False)
    
    def __post_init__(self):
        n = self.max_agents
        self.positions = np.zeros((n, 2), dtype=np.float64)
        self.velocities = np.zeros((n, 2), dtype=np.float64)
        self.target_velocities = np.zeros((n, 2), dtype=np.float64)
        
        self.ids = np.zeros(n, dtype=np.int32)
        self.patience = np.ones(n, dtype=np.float64)
        self.max_speeds = np.full(n, MAX_SPEED, dtype=np.float64)
        self.radii = np.full(n, AGENT_RADIUS, dtype=np.float64)
        self.states = np.zeros(n, dtype=np.int32)
        
        self.waypoints = [[] for _ in range(n)]
        self.waypoint_indices = np.zeros(n, dtype=np.int32)
        
        self.active = np.zeros(n, dtype=bool)
    
    def spawn(
        self,
        position: Vec2,
        waypoints: List[Vec2],
        velocity: Optional[Vec2] = None,
        max_speed: float = MAX_SPEED
    ) -> int:
        """Spawn a new agent. Returns agent index or -1 if pool is full."""
        # Find first inactive slot
        inactive = np.where(~self.active)[0]
        if len(inactive) == 0:
            return -1
        
        idx = inactive[0]
        
        self.ids[idx] = self._next_id
        self._next_id += 1
        
        self.positions[idx] = position
        self.velocities[idx] = velocity if velocity is not None else zero()
        self.target_velocities[idx] = zero()
        
        self.patience[idx] = 1.0
        self.max_speeds[idx] = max_speed
        self.radii[idx] = AGENT_RADIUS
        self.states[idx] = 0
        
        self.waypoints[idx] = list(waypoints)
        self.waypoint_indices[idx] = 0
        
        self.active[idx] = True
        self.count += 1
        
        return idx
    
    def despawn(self, idx: int):
        """Remove agent at index."""
        if self.active[idx]:
            self.active[idx] = False
            self.waypoints[idx] = []
            self.count -= 1
    
    def get_active_indices(self) -> np.ndarray:
        """Get indices of all active agents."""
        return np.where(self.active)[0]
    
    def get_positions(self) -> np.ndarray:
        """Get positions of active agents only."""
        return self.positions[self.active]
    
    def get_velocities(self) -> np.ndarray:
        """Get velocities of active agents only."""
        return self.velocities[self.active]


# ============================================================================
# Steering Behaviors
# ============================================================================

def compute_path_following_force(
    pool: AgentPool,
    idx: int
) -> Vec2:
    """
    Compute steering force to follow waypoint path.
    Uses arrival behavior near waypoints.
    """
    waypoints = pool.waypoints[idx]
    if not waypoints:
        return zero()
    
    wp_idx = pool.waypoint_indices[idx]
    if wp_idx >= len(waypoints):
        return zero()
    
    target = waypoints[wp_idx]
    position = pool.positions[idx]
    
    to_target = target - position
    dist = magnitude(to_target)
    
    # Check if arrived at waypoint
    if dist < WAYPOINT_ARRIVAL_DIST:
        pool.waypoint_indices[idx] = wp_idx + 1
        # Recurse for next waypoint
        return compute_path_following_force(pool, idx)
    
    # Desired velocity toward target
    max_speed = pool.max_speeds[idx]
    
    # Arrival behavior: slow down near waypoint
    if dist < SLOWDOWN_DIST:
        desired_speed = max_speed * (dist / SLOWDOWN_DIST)
    else:
        desired_speed = max_speed
    
    desired_velocity = normalize(to_target) * desired_speed
    
    # Steering = desired - current
    current_velocity = pool.velocities[idx]
    steering = desired_velocity - current_velocity
    
    return steering


def compute_separation_force(
    pool: AgentPool,
    idx: int,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray
) -> Vec2:
    """
    Compute force to maintain personal space.
    Inversely proportional to distance.
    """
    if len(neighbor_indices) == 0:
        return zero()
    
    position = pool.positions[idx]
    force = zero()
    
    for ni, dist in zip(neighbor_indices, neighbor_distances):
        if ni == idx:
            continue
        
        if dist < 0.001:  # Overlapping
            # Random push
            force += random_in_circle(1.0)
        elif dist < PERSONAL_SPACE:
            # Push away, stronger when closer
            away = position - pool.positions[ni]
            strength = (PERSONAL_SPACE - dist) / PERSONAL_SPACE
            force += normalize(away) * strength
    
    return force


def compute_collision_avoidance_force(
    pool: AgentPool,
    idx: int,
    neighbor_indices: np.ndarray
) -> Vec2:
    """
    Predictive collision avoidance.
    Looks ahead and steers to avoid future collisions.
    """
    if len(neighbor_indices) == 0:
        return zero()
    
    position = pool.positions[idx]
    velocity = pool.velocities[idx]
    radius = pool.radii[idx]
    
    most_threatening = None
    min_time = float('inf')
    
    for ni in neighbor_indices:
        if ni == idx:
            continue
        
        other_pos = pool.positions[ni]
        other_vel = pool.velocities[ni]
        other_radius = pool.radii[ni]
        
        # Relative position and velocity
        rel_pos = other_pos - position
        rel_vel = other_vel - velocity
        
        # Time to closest approach
        rel_speed_sq = dot(rel_vel, rel_vel)
        if rel_speed_sq < 0.0001:
            continue
        
        t = -dot(rel_pos, rel_vel) / rel_speed_sq
        
        if t < 0 or t > COLLISION_HORIZON:
            continue
        
        # Distance at closest approach
        closest_dist = magnitude(rel_pos + rel_vel * t)
        combined_radius = radius + other_radius + 0.2  # Buffer
        
        if closest_dist < combined_radius and t < min_time:
            min_time = t
            most_threatening = ni
    
    if most_threatening is None:
        return zero()
    
    # Steer away from collision point
    threat_pos = pool.positions[most_threatening]
    threat_vel = pool.velocities[most_threatening]
    
    future_pos = position + velocity * min_time
    future_threat = threat_pos + threat_vel * min_time
    
    avoidance = future_pos - future_threat
    
    if magnitude(avoidance) < 0.001:
        # Head-on, dodge sideways
        avoidance = vec2(-velocity[1], velocity[0])
    
    # Urgency based on time
    urgency = 1.0 - (min_time / COLLISION_HORIZON)
    
    return normalize(avoidance) * urgency


def compute_alignment_force(
    pool: AgentPool,
    idx: int,
    neighbor_indices: np.ndarray
) -> Vec2:
    """
    Align with average velocity of neighbors.
    Creates lane formation in crowds.
    """
    if len(neighbor_indices) == 0:
        return zero()
    
    avg_velocity = zero()
    count = 0
    
    for ni in neighbor_indices:
        if ni == idx:
            continue
        vel = pool.velocities[ni]
        if magnitude(vel) > MIN_SPEED:
            avg_velocity = avg_velocity + vel
            count += 1
    
    if count == 0:
        return zero()
    
    avg_velocity = avg_velocity / count
    
    # Steer toward average
    steering = avg_velocity - pool.velocities[idx]
    return steering * 0.5  # Gentle influence


# ============================================================================
# Spatial Hashing for Neighbor Queries
# ============================================================================

class SpatialHash:
    """Grid-based spatial hashing for O(1) neighbor queries."""
    
    def __init__(self, cell_size: float = 3.0):
        self.cell_size = cell_size
        self.grid: dict[Tuple[int, int], List[int]] = {}
    
    def _cell(self, pos: Vec2) -> Tuple[int, int]:
        return (int(pos[0] // self.cell_size), int(pos[1] // self.cell_size))
    
    def rebuild(self, pool: AgentPool):
        """Rebuild spatial hash from agent pool."""
        self.grid.clear()
        
        for idx in pool.get_active_indices():
            cell = self._cell(pool.positions[idx])
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(idx)
    
    def query_radius(
        self,
        position: Vec2,
        radius: float,
        pool: AgentPool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find all agents within radius of position.
        Returns (indices, distances).
        """
        cell = self._cell(position)
        cells_to_check = int(np.ceil(radius / self.cell_size)) + 1
        
        candidates = []
        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                check_cell = (cell[0] + dx, cell[1] + dy)
                if check_cell in self.grid:
                    candidates.extend(self.grid[check_cell])
        
        if not candidates:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
        
        candidate_arr = np.array(candidates, dtype=np.int32)
        candidate_positions = pool.positions[candidate_arr]
        
        distances = batch_distances(candidate_positions, position)
        mask = distances <= radius
        
        return candidate_arr[mask], distances[mask]


# ============================================================================
# Agent Update
# ============================================================================

def classify_agent_state(speed: float, patience: float) -> int:
    """Classify agent state based on speed and patience."""
    if speed < MIN_SPEED:
        if patience < 0.3:
            return 3  # pushing
        return 2  # stopped
    elif speed < MAX_SPEED * 0.5:
        return 1  # slowing
    else:
        return 0  # moving


def update_agents(
    pool: AgentPool,
    spatial_hash: SpatialHash,
    dt: float
):
    """
    Update all agents for one simulation tick.
    """
    # Rebuild spatial hash
    spatial_hash.rebuild(pool)
    
    active_indices = pool.get_active_indices()
    
    # Compute forces for all active agents
    for idx in active_indices:
        position = pool.positions[idx]
        
        # Get neighbors
        neighbor_indices, neighbor_distances = spatial_hash.query_radius(
            position, NEIGHBOR_RADIUS, pool
        )
        
        # Compute steering forces
        path_force = compute_path_following_force(pool, idx)
        separation_force = compute_separation_force(pool, idx, neighbor_indices, neighbor_distances)
        avoidance_force = compute_collision_avoidance_force(pool, idx, neighbor_indices)
        alignment_force = compute_alignment_force(pool, idx, neighbor_indices)
        
        # Combine forces
        total_force = (
            path_force * PATH_WEIGHT +
            separation_force * SEPARATION_WEIGHT +
            avoidance_force * AVOIDANCE_WEIGHT +
            alignment_force * ALIGNMENT_WEIGHT
        )
        
        # Clamp acceleration
        acceleration = clamp_magnitude(total_force, MAX_ACCELERATION)
        
        # Update velocity with time-based friction (frame-rate independent)
        friction = np.exp(-FRICTION_DECAY * dt)  # Smooth decay
        new_velocity = pool.velocities[idx] * friction + acceleration * dt
        new_velocity = clamp_magnitude(new_velocity, pool.max_speeds[idx])
        
        pool.velocities[idx] = new_velocity
        
        # Update position
        pool.positions[idx] = pool.positions[idx] + new_velocity * dt
        
        # Update patience (decreases when blocked)
        speed = magnitude(new_velocity)
        if speed < MIN_SPEED:
            pool.patience[idx] = max(0, pool.patience[idx] - 0.01 * dt)
        else:
            pool.patience[idx] = min(1.0, pool.patience[idx] + 0.05 * dt)
        
        # Update state
        pool.states[idx] = classify_agent_state(speed, pool.patience[idx])
    
    # Remove agents that completed their path
    for idx in active_indices:
        wp_idx = pool.waypoint_indices[idx]
        if wp_idx >= len(pool.waypoints[idx]):
            pool.despawn(idx)


# ============================================================================
# Spawn Management
# ============================================================================

@dataclass
class SpawnPoint:
    """Entry point for agents."""
    position: Vec2
    direction: float  # radians
    waypoints: List[Vec2]  # path to follow


def spawn_agents_at_rate(
    pool: AgentPool,
    spawn_points: List[SpawnPoint],
    rate: float,  # agents per second
    dt: float,
    rng: np.random.Generator
) -> int:
    """
    Probabilistically spawn agents based on rate.
    Returns number spawned.
    """
    expected = rate * dt
    
    # Poisson-ish: spawn floor(expected) + maybe one more
    count = int(expected)
    if rng.random() < (expected - count):
        count += 1
    
    spawned = 0
    for _ in range(count):
        if len(spawn_points) == 0:
            break
        
        sp = spawn_points[rng.integers(0, len(spawn_points))]
        
        # Randomize spawn position slightly
        offset = random_in_circle(1.0)
        position = sp.position + offset
        
        # Randomize speed slightly
        max_speed = MAX_SPEED * (0.8 + rng.random() * 0.4)
        
        # Initial velocity in spawn direction
        initial_vel = from_angle(sp.direction, max_speed * 0.5)
        
        idx = pool.spawn(position, sp.waypoints, initial_vel, max_speed)
        if idx >= 0:
            spawned += 1
    
    return spawned
