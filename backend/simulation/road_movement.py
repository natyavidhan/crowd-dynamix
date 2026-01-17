"""
Road-Constrained Agent Movement System.
Agents MUST stay on roads at all times. They follow road centerlines
with lateral offset for side-by-side walking.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vector import Vec2, vec2, magnitude, normalize, distance, zero


# ============================================================================
# Constants
# ============================================================================

# Movement
DEFAULT_WALK_SPEED = 1.2       # m/s normal walking
SPEED_VARIANCE = 0.3           # ± variance in speed
MIN_SPEED = 0.1                # m/s minimum movement
ACCELERATION = 2.0             # m/s² how fast to reach target speed

# Road constraints
LANE_WIDTH = 0.6               # meters per lane (agent width + buffer)
AGENT_RADIUS = 0.25            # meters
SEPARATION_DIST = 0.5          # minimum distance between agents

# Navigation
EXIT_ARRIVAL_DIST = 3.0        # meters - when to consider "at exit"
ROAD_CONNECTION_DIST = 5.0     # meters - max gap between road endpoints to consider connected


# ============================================================================
# Road Agent State
# ============================================================================

@dataclass
class RoadAgent:
    """
    Agent constrained to road network.
    Position is defined by (road_id, distance_along_road, lateral_offset).
    """
    id: int
    road_id: str
    distance_along: float       # meters from road start
    lateral_offset: float       # meters from centerline (-width/2 to +width/2)
    speed: float                # current speed m/s
    target_speed: float         # desired speed m/s
    
    # Navigation
    direction: int = 1          # 1 = forward along road, -1 = backward
    target_exit_id: Optional[str] = None
    path: List[str] = field(default_factory=list)  # sequence of road IDs to follow
    path_index: int = 0
    
    # State
    active: bool = True
    waiting: bool = False       # stuck behind others
    # Distance traveled since last road transition or spawn. Used to avoid
    # immediate exit at junctions (agents must travel a bit into the road
    # before being allowed to exit).
    travel_since_transition: float = 0.0
    
    def get_world_position(self, roads: Dict[str, 'RoadSegment']) -> Vec2:
        """Convert road position to world coordinates."""
        road = roads.get(self.road_id)
        if not road:
            return zero()
        
        # Get centerline position
        center = road.point_at_distance(self.distance_along)
        
        # Get road direction at this point for perpendicular offset
        direction = road.direction_at_distance(self.distance_along)
        
        # Perpendicular vector (rotate 90 degrees)
        perp = vec2(-direction[1], direction[0])
        
        # Apply lateral offset
        return center + perp * self.lateral_offset


# ============================================================================
# Road Segment (simplified for movement)
# ============================================================================

@dataclass
class RoadSegment:
    """Road segment with movement properties."""
    id: str
    name: str
    width: float                    # road width in meters
    speed_limit: float              # max speed on this road
    bidirectional: bool
    centerline: List[Vec2]          # list of points defining the road
    
    # Precomputed
    length: float = field(init=False)
    segments: List[Tuple[Vec2, Vec2, float]] = field(init=False)
    
    # Connections at endpoints
    connections_start: List[str] = field(default_factory=list)  # roads connected at start
    connections_end: List[str] = field(default_factory=list)    # roads connected at end
    
    # Mid-road junctions: (distance_along, connected_road_id)
    # These are T-junctions where another road connects along this road's length
    mid_junctions: List[Tuple[float, str]] = field(default_factory=list)
    
    # Exit info
    has_exit_at_start: bool = False
    has_exit_at_end: bool = False
    exit_id_start: Optional[str] = None
    exit_id_end: Optional[str] = None
    
    def __post_init__(self):
        self.segments = []
        self.length = 0.0
        
        for i in range(len(self.centerline) - 1):
            start = self.centerline[i]
            end = self.centerline[i + 1]
            seg_len = magnitude(end - start)
            self.segments.append((start, end, seg_len))
            self.length += seg_len
    
    def point_at_distance(self, dist: float) -> Vec2:
        """Get world position at distance along road."""
        dist = max(0, min(dist, self.length))
        
        cumulative = 0.0
        for start, end, seg_len in self.segments:
            if cumulative + seg_len >= dist:
                t = (dist - cumulative) / seg_len if seg_len > 0 else 0
                return start + (end - start) * t
            cumulative += seg_len
        
        return self.centerline[-1].copy()
    
    def direction_at_distance(self, dist: float) -> Vec2:
        """Get road direction at distance."""
        cumulative = 0.0
        for start, end, seg_len in self.segments:
            if cumulative + seg_len >= dist:
                if seg_len > 0:
                    return normalize(end - start)
                break
            cumulative += seg_len
        
        # Fallback to last segment direction
        if self.segments:
            start, end, _ = self.segments[-1]
            return normalize(end - start)
        return vec2(1, 0)
    
    def get_lane_count(self) -> int:
        """How many agents can walk side-by-side."""
        return max(1, int(self.width / LANE_WIDTH))
    
    def get_lane_offset(self, lane: int) -> float:
        """Get lateral offset for a specific lane."""
        lane_count = self.get_lane_count()
        if lane_count == 1:
            return 0.0
        
        # Distribute lanes evenly across width
        # Lane 0 is leftmost, lane N-1 is rightmost
        total_width = (lane_count - 1) * LANE_WIDTH
        return -total_width / 2 + lane * LANE_WIDTH


# ============================================================================
# Road Network Manager
# ============================================================================

@dataclass
class RoadNetworkManager:
    """Manages road network and agent navigation."""
    roads: Dict[str, RoadSegment] = field(default_factory=dict)
    exits: Dict[str, Vec2] = field(default_factory=dict)
    
    # Precomputed paths from each road to each exit
    path_cache: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    
    def add_road(self, road: RoadSegment):
        """Add a road to the network."""
        self.roads[road.id] = road
    
    def add_exit(self, exit_id: str, position: Vec2):
        """Add an exit point."""
        self.exits[exit_id] = position
    
    def build_connections(self):
        """Build road-to-road connections and road-to-exit connections.
        
        Supports T-junctions where one road's endpoint connects to any point
        along another road (not just endpoints).
        """
        road_list = list(self.roads.values())
        
        # Initialize connection lists
        for road in road_list:
            road.connections_start = []
            road.connections_end = []
            road.mid_junctions = []
        
        # Find road-to-road connections
        for i, road1 in enumerate(road_list):
            start1 = road1.centerline[0]
            end1 = road1.centerline[-1]
            
            for j, road2 in enumerate(road_list):
                if i == j:
                    continue
                
                start2 = road2.centerline[0]
                end2 = road2.centerline[-1]
                
                # Check endpoint-to-endpoint connections
                if distance(end1, start2) < ROAD_CONNECTION_DIST:
                    if road2.id not in road1.connections_end:
                        road1.connections_end.append(road2.id)
                    if road1.id not in road2.connections_start:
                        road2.connections_start.append(road1.id)
                        
                if distance(end1, end2) < ROAD_CONNECTION_DIST:
                    if road2.id not in road1.connections_end:
                        road1.connections_end.append(road2.id)
                    if road1.id not in road2.connections_end:
                        road2.connections_end.append(road1.id)
                        
                if distance(start1, start2) < ROAD_CONNECTION_DIST:
                    if road2.id not in road1.connections_start:
                        road1.connections_start.append(road2.id)
                    if road1.id not in road2.connections_start:
                        road2.connections_start.append(road1.id)
                        
                if distance(start1, end2) < ROAD_CONNECTION_DIST:
                    if road2.id not in road1.connections_start:
                        road1.connections_start.append(road2.id)
                    if road1.id not in road2.connections_end:
                        road2.connections_end.append(road1.id)
                
                # Check T-junction: road1's endpoint near any point along road2
                # This handles cases where access roads connect to main road
                _, dist_along1, perp_dist1 = self._nearest_point_on_road(road2, end1)
                if perp_dist1 < ROAD_CONNECTION_DIST:
                    if road2.id not in road1.connections_end:
                        road1.connections_end.append(road2.id)
                    
                    # Check if this is a mid-road junction (not at road2's endpoints)
                    is_mid_junction = (dist_along1 > ROAD_CONNECTION_DIST and 
                                       dist_along1 < road2.length - ROAD_CONNECTION_DIST)
                    if is_mid_junction:
                        # Add mid-junction to road2: at dist_along1, connects to road1
                        road2.mid_junctions.append((dist_along1, road1.id))
                    else:
                        # It's an endpoint connection
                        if dist_along1 < road2.length / 2:
                            if road1.id not in road2.connections_start:
                                road2.connections_start.append(road1.id)
                        else:
                            if road1.id not in road2.connections_end:
                                road2.connections_end.append(road1.id)
                
                _, dist_along2, perp_dist2 = self._nearest_point_on_road(road2, start1)
                if perp_dist2 < ROAD_CONNECTION_DIST:
                    if road2.id not in road1.connections_start:
                        road1.connections_start.append(road2.id)
                    
                    # Check if this is a mid-road junction
                    is_mid_junction = (dist_along2 > ROAD_CONNECTION_DIST and 
                                       dist_along2 < road2.length - ROAD_CONNECTION_DIST)
                    if is_mid_junction:
                        road2.mid_junctions.append((dist_along2, road1.id))
                    else:
                        if dist_along2 < road2.length / 2:
                            if road1.id not in road2.connections_start:
                                road2.connections_start.append(road1.id)
                        else:
                            if road1.id not in road2.connections_end:
                                road2.connections_end.append(road1.id)
        
        # Sort mid-junctions by distance along road
        for road in road_list:
            road.mid_junctions.sort(key=lambda x: x[0])
        
        # Find road-to-exit connections
        # Mark exits at road endpoints regardless of whether there are road connections
        # (A junction can also be an exit)
        EXIT_PROXIMITY = 20.0  # Larger radius to catch exits near road ends
        
        for road in road_list:
            start = road.centerline[0]
            end = road.centerline[-1]
            
            for exit_id, exit_pos in self.exits.items():
                # Check exit near start
                if distance(start, exit_pos) < EXIT_PROXIMITY:
                    road.has_exit_at_start = True
                    road.exit_id_start = exit_id
                
                # Check exit near end
                if distance(end, exit_pos) < EXIT_PROXIMITY:
                    road.has_exit_at_end = True
                    road.exit_id_end = exit_id
                
                # Also check if exit is along the general direction at end of road
                # This helps with exits that are a bit beyond the road end
                if not road.has_exit_at_end:
                    road_dir = road.direction_at_distance(road.length)
                    to_exit = exit_pos - end
                    to_exit_dist = magnitude(to_exit)
                    if to_exit_dist > 0.1 and to_exit_dist < 50.0:  # Within 50m
                        to_exit_norm = to_exit / to_exit_dist
                        dot = np.dot(road_dir, to_exit_norm)
                        if dot > 0.7:  # Exit is roughly in direction of road end
                            road.has_exit_at_end = True
                            road.exit_id_end = exit_id
                
                # Same for start, but check opposite direction
                if not road.has_exit_at_start:
                    road_dir = road.direction_at_distance(0)
                    to_exit = exit_pos - start
                    to_exit_dist = magnitude(to_exit)
                    if to_exit_dist > 0.1 and to_exit_dist < 50.0:
                        to_exit_norm = to_exit / to_exit_dist
                        dot = np.dot(road_dir, to_exit_norm)
                        if dot < -0.7:  # Exit is opposite to road direction at start
                            road.has_exit_at_start = True
                            road.exit_id_start = exit_id
        
        # Build path cache using BFS
        self._build_path_cache()
    
    def _build_path_cache(self):
        """Precompute shortest paths from each road to each exit."""
        self.path_cache.clear()
        
        for exit_id in self.exits:
            # Find all roads that lead to this exit
            exit_roads = []
            for road in self.roads.values():
                if road.exit_id_start == exit_id or road.exit_id_end == exit_id:
                    exit_roads.append(road.id)
            
            if not exit_roads:
                continue
            
            # BFS from exit roads to all other roads
            visited: Set[str] = set()
            parent: Dict[str, Tuple[str, str]] = {}  # road_id -> (parent_road_id, "start"|"end")
            queue = list(exit_roads)
            
            for road_id in exit_roads:
                visited.add(road_id)
                parent[road_id] = (road_id, "self")
            
            while queue:
                current_id = queue.pop(0)
                current = self.roads[current_id]
                
                # Check connections at start
                for conn_id in current.connections_start:
                    if conn_id not in visited:
                        visited.add(conn_id)
                        parent[conn_id] = (current_id, "start")
                        queue.append(conn_id)
                
                # Check connections at end
                for conn_id in current.connections_end:
                    if conn_id not in visited:
                        visited.add(conn_id)
                        parent[conn_id] = (current_id, "end")
                        queue.append(conn_id)
            
            # Build paths for each road to this exit
            for road_id in self.roads:
                if road_id in visited:
                    path = []
                    current = road_id
                    while parent[current][1] != "self":
                        path.append(current)
                        current = parent[current][0]
                    path.append(current)
                    self.path_cache[(road_id, exit_id)] = path
    
    def get_path_to_exit(self, road_id: str, exit_id: str) -> List[str]:
        """Get precomputed path from road to exit."""
        return self.path_cache.get((road_id, exit_id), [])
    
    def find_nearest_road(self, pos: Vec2) -> Optional[Tuple[str, float, float]]:
        """
        Find nearest road to a position.
        Returns (road_id, distance_along, perpendicular_distance) or None.
        """
        best_road_id = None
        best_dist_along = 0.0
        min_perp_dist = float('inf')
        
        for road_id, road in self.roads.items():
            point, dist_along, perp_dist = self._nearest_point_on_road(road, pos)
            if perp_dist < min_perp_dist:
                min_perp_dist = perp_dist
                best_road_id = road_id
                best_dist_along = dist_along
        
        if best_road_id:
            return best_road_id, best_dist_along, min_perp_dist
        return None
    
    def _nearest_point_on_road(self, road: RoadSegment, pos: Vec2) -> Tuple[Vec2, float, float]:
        """Find nearest point on a road segment."""
        min_dist = float('inf')
        best_point = road.centerline[0]
        best_dist_along = 0.0
        
        cumulative = 0.0
        for start, end, seg_len in road.segments:
            if seg_len < 0.001:
                cumulative += seg_len
                continue
            
            seg_vec = end - start
            pos_vec = pos - start
            t = np.dot(pos_vec, seg_vec) / (seg_len * seg_len)
            t = max(0, min(1, t))
            
            proj = start + seg_vec * t
            d = magnitude(pos - proj)
            
            if d < min_dist:
                min_dist = d
                best_point = proj
                best_dist_along = cumulative + t * seg_len
            
            cumulative += seg_len
        
        return best_point, best_dist_along, min_dist


# ============================================================================
# Road-Constrained Agent Pool
# ============================================================================

@dataclass
class RoadAgentPool:
    """Pool of road-constrained agents."""
    max_agents: int
    network: RoadNetworkManager
    
    agents: Dict[int, RoadAgent] = field(default_factory=dict)
    _next_id: int = 0
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    
    def spawn(
        self,
        road_id: str,
        distance_along: float = 0.0,
        target_exit_id: Optional[str] = None,
        lateral_offset: Optional[float] = None
    ) -> Optional[int]:
        """Spawn an agent on a road."""
        if len(self.agents) >= self.max_agents:
            return None
        
        road = self.network.roads.get(road_id)
        if not road:
            return None
        
        # Assign lateral offset (lane)
        if lateral_offset is None:
            # Find an available lane
            lane_count = road.get_lane_count()
            lane = self._rng.integers(0, lane_count)
            lateral_offset = road.get_lane_offset(lane)
        
        # Clamp lateral offset to road width
        max_offset = (road.width / 2) - AGENT_RADIUS
        lateral_offset = max(-max_offset, min(max_offset, lateral_offset))
        
        # Determine target speed with variance
        target_speed = DEFAULT_WALK_SPEED + self._rng.uniform(-SPEED_VARIANCE, SPEED_VARIANCE)
        target_speed = min(target_speed, road.speed_limit)
        
        # Determine direction based on path
        direction = 1
        path = []
        if target_exit_id:
            path = self.network.get_path_to_exit(road_id, target_exit_id)
            if path and len(path) > 1:
                # Determine which direction to go on this road
                next_road_id = path[1] if len(path) > 1 else path[0]
                road_obj = road
                
                # Check if next road connects at start or end
                if next_road_id in road_obj.connections_end:
                    direction = 1  # Go toward end
                elif next_road_id in road_obj.connections_start:
                    direction = -1  # Go toward start
                else:
                    # Check if next road is a mid-junction
                    for junc_dist, junc_road_id in road_obj.mid_junctions:
                        if junc_road_id == next_road_id:
                            # Need to go toward this junction point
                            if junc_dist > distance_along:
                                direction = 1  # Junction is ahead
                            else:
                                direction = -1  # Junction is behind
                            break
                    else:
                        # Fall back to exit-based direction
                        if road_obj.exit_id_end == target_exit_id:
                            direction = 1
                        elif road_obj.exit_id_start == target_exit_id:
                            direction = -1
        
        agent = RoadAgent(
            id=self._next_id,
            road_id=road_id,
            distance_along=distance_along,
            lateral_offset=lateral_offset,
            speed=target_speed * 0.5,  # Start slower
            target_speed=target_speed,
            direction=direction,
            target_exit_id=target_exit_id,
            path=path,
            path_index=0
        )
        
        self.agents[self._next_id] = agent
        self._next_id += 1
        
        return agent.id
    
    def despawn(self, agent_id: int):
        """Remove an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def get_active_count(self) -> int:
        return len(self.agents)
    
    def get_world_positions(self) -> List[Tuple[int, Vec2]]:
        """Get world positions of all agents."""
        result = []
        for agent in self.agents.values():
            if agent.active:
                pos = agent.get_world_position(self.network.roads)
                result.append((agent.id, pos))
        return result


# ============================================================================
# Movement Update
# ============================================================================

def update_road_agents(
    pool: RoadAgentPool,
    dt: float
):
    """Update all road-constrained agents."""
    # Build spatial index of agents on each road
    agents_on_road: Dict[str, List[RoadAgent]] = {}
    for agent in pool.agents.values():
        if agent.road_id not in agents_on_road:
            agents_on_road[agent.road_id] = []
        agents_on_road[agent.road_id].append(agent)
    
    # Sort agents on each road by distance (for collision detection)
    for road_id in agents_on_road:
        agents_on_road[road_id].sort(key=lambda a: a.distance_along)
    
    agents_to_remove = []
    
    for agent in list(pool.agents.values()):
        if not agent.active:
            continue
        
        road = pool.network.roads.get(agent.road_id)
        if not road:
            agents_to_remove.append(agent.id)
            continue
        
        # Check for agents ahead (same road, same direction)
        blocked = False
        road_agents = agents_on_road.get(agent.road_id, [])
        
        for other in road_agents:
            if other.id == agent.id:
                continue
            
            # Check if other agent is ahead
            if agent.direction > 0:
                # Moving forward, check agents ahead
                dist_diff = other.distance_along - agent.distance_along
            else:
                # Moving backward, check agents behind (lower distance)
                dist_diff = agent.distance_along - other.distance_along
            
            # Only consider agents in front and in same-ish lane
            lateral_diff = abs(other.lateral_offset - agent.lateral_offset)
            
            if 0 < dist_diff < SEPARATION_DIST * 2 and lateral_diff < LANE_WIDTH:
                # Agent ahead - slow down or stop
                if dist_diff < SEPARATION_DIST:
                    blocked = True
                else:
                    # Slow down to match
                    agent.target_speed = min(agent.target_speed, other.speed * 0.9)
        
        # Update speed
        if blocked:
            agent.speed = max(0, agent.speed - ACCELERATION * dt * 2)
            agent.waiting = True
        else:
            agent.waiting = False
            target = min(agent.target_speed, road.speed_limit)
            if agent.speed < target:
                agent.speed = min(target, agent.speed + ACCELERATION * dt)
            elif agent.speed > target:
                agent.speed = max(target, agent.speed - ACCELERATION * dt)
        
        # Move along road
        movement = agent.speed * dt * agent.direction
        prev_dist = agent.distance_along
        agent.distance_along += movement
        # Accumulate travel since last transition/spawn
        agent.travel_since_transition += abs(movement)
        
        # Check for mid-road junctions (T-junctions)
        # Agent should turn if it passes a junction that leads to its target
        if road.mid_junctions and agent.path and len(agent.path) > 1:
            next_road_in_path = agent.path[agent.path_index + 1] if agent.path_index + 1 < len(agent.path) else None
            
            for junc_dist, junc_road_id in road.mid_junctions:
                # Check if agent crossed this junction
                if agent.direction > 0:
                    crossed = prev_dist < junc_dist <= agent.distance_along
                else:
                    crossed = agent.distance_along <= junc_dist < prev_dist
                
                if crossed and junc_road_id == next_road_in_path:
                    # Agent should turn here
                    _transition_at_junction(agent, junc_road_id, pool.network, junc_dist)
                    break
        
        # Check if reached end of road
        if agent.distance_along >= road.length:
            # Reached end
            # Only exit if this is the agent's target exit
            # Require a minimum distance traveled on this road before exiting
            MIN_EXIT_TRAVEL = 2.0
            
            # Debug
            # print(f"DEBUG: Agent at road end. has_exit_at_end={road.has_exit_at_end}, exit_id_end={road.exit_id_end}, target={agent.target_exit_id}, travel={agent.travel_since_transition}")
            
            if (
                road.has_exit_at_end
                and agent.target_exit_id == road.exit_id_end
                and agent.travel_since_transition >= MIN_EXIT_TRAVEL
            ):
                # Reached target exit - remove agent
                agents_to_remove.append(agent.id)
                continue
            
            # Need to transition to next road
            next_road_id = _find_next_road(agent, road, pool.network, at_end=True)
            if next_road_id:
                _transition_to_road(agent, next_road_id, pool.network, from_end=True)
            else:
                # Dead end - turn around if bidirectional, or exit here if any exit
                if road.has_exit_at_end and agent.travel_since_transition >= MIN_EXIT_TRAVEL:
                    agents_to_remove.append(agent.id)
                elif road.bidirectional:
                    agent.direction = -1
                    agent.distance_along = road.length
                else:
                    agents_to_remove.append(agent.id)
        
        elif agent.distance_along <= 0:
            # Reached start
            # Only exit if this is the agent's target exit
            # Require a minimum distance traveled on this road before exiting
            MIN_EXIT_TRAVEL = 2.0
            if (
                road.has_exit_at_start
                and agent.target_exit_id == road.exit_id_start
                and agent.travel_since_transition >= MIN_EXIT_TRAVEL
            ):
                # Reached target exit - remove agent
                agents_to_remove.append(agent.id)
                continue
            
            # Need to transition to next road
            next_road_id = _find_next_road(agent, road, pool.network, at_end=False)
            if next_road_id:
                _transition_to_road(agent, next_road_id, pool.network, from_end=False)
            else:
                # Dead end - turn around if bidirectional, or exit here if any exit
                if road.has_exit_at_start and agent.travel_since_transition >= MIN_EXIT_TRAVEL:
                    agents_to_remove.append(agent.id)
                elif road.bidirectional:
                    agent.direction = 1
                    agent.distance_along = 0
                else:
                    agents_to_remove.append(agent.id)
    
    # Remove finished agents
    for agent_id in agents_to_remove:
        pool.despawn(agent_id)


def _find_next_road(
    agent: RoadAgent,
    current_road: RoadSegment,
    network: RoadNetworkManager,
    at_end: bool
) -> Optional[str]:
    """Find the next road to transition to."""
    # First check if we have a planned path
    if agent.path and agent.path_index < len(agent.path) - 1:
        next_in_path = agent.path[agent.path_index + 1]
        connections = current_road.connections_end if at_end else current_road.connections_start
        if next_in_path in connections:
            return next_in_path
    
    # Otherwise find any connected road that leads toward exit
    connections = current_road.connections_end if at_end else current_road.connections_start
    
    if not connections:
        return None
    
    # If we have a target exit, find road that leads there
    if agent.target_exit_id:
        best_road = None
        best_path_len = float('inf')
        
        for conn_id in connections:
            path = network.get_path_to_exit(conn_id, agent.target_exit_id)
            if path and len(path) < best_path_len:
                best_path_len = len(path)
                best_road = conn_id
        
        if best_road:
            return best_road
    
    # Just take any connection
    return connections[0] if connections else None


def _transition_at_junction(
    agent: RoadAgent,
    new_road_id: str,
    network: RoadNetworkManager,
    junction_dist: float
):
    """Transition agent at a mid-road junction point."""
    new_road = network.roads.get(new_road_id)
    if not new_road:
        return
    
    old_road = network.roads.get(agent.road_id)
    if not old_road:
        return
    
    # Get the junction point on the old road
    junction_point = old_road.point_at_distance(junction_dist)
    
    # Find which end of new road is closest to the junction point
    dist_to_start = distance(junction_point, new_road.centerline[0])
    dist_to_end = distance(junction_point, new_road.centerline[-1])
    
    if dist_to_start < dist_to_end:
        # Enter at start, go forward
        agent.distance_along = 0
        agent.direction = 1
    else:
        # Enter at end, go backward
        agent.distance_along = new_road.length
        agent.direction = -1
    
    agent.road_id = new_road_id
    
    # Update path index
    if agent.path:
        try:
            agent.path_index = agent.path.index(new_road_id)
        except ValueError:
            # Road not in path, recalculate
            if agent.target_exit_id:
                agent.path = network.get_path_to_exit(new_road_id, agent.target_exit_id)
                agent.path_index = 0
    
    # Adjust lateral offset for new road width
    max_offset = (new_road.width / 2) - AGENT_RADIUS
    agent.lateral_offset = max(-max_offset, min(max_offset, agent.lateral_offset))
    # Reset travel accumulator so agent must travel into the new road before exiting
    agent.travel_since_transition = 0.0


def _transition_to_road(
    agent: RoadAgent,
    new_road_id: str,
    network: RoadNetworkManager,
    from_end: bool
):
    """Transition agent to a new road."""
    new_road = network.roads.get(new_road_id)
    if not new_road:
        return
    
    old_road = network.roads.get(agent.road_id)
    if not old_road:
        return
    
    # Determine entry point and direction on new road
    old_endpoint = old_road.centerline[-1] if from_end else old_road.centerline[0]
    
    # Find which end of new road is closest
    dist_to_start = distance(old_endpoint, new_road.centerline[0])
    dist_to_end = distance(old_endpoint, new_road.centerline[-1])
    
    if dist_to_start < dist_to_end:
        # Enter at start, go forward
        agent.distance_along = 0
        agent.direction = 1
    else:
        # Enter at end, go backward
        agent.distance_along = new_road.length
        agent.direction = -1
    
    agent.road_id = new_road_id
    
    # Update path index
    if agent.path:
        try:
            agent.path_index = agent.path.index(new_road_id)
        except ValueError:
            # Road not in path, recalculate
            if agent.target_exit_id:
                agent.path = network.get_path_to_exit(new_road_id, agent.target_exit_id)
                agent.path_index = 0
    
    # Adjust lateral offset for new road width
    max_offset = (new_road.width / 2) - AGENT_RADIUS
    agent.lateral_offset = max(-max_offset, min(max_offset, agent.lateral_offset))
    # Reset travel accumulator so agent must travel into the new road before exiting
    agent.travel_since_transition = 0.0


# ============================================================================
# Spawn Management
# ============================================================================

@dataclass
class RoadSpawnPoint:
    """Spawn point on a road."""
    id: str
    road_id: str
    distance_along: float
    target_exit_id: str
    rate: float  # agents per second


def spawn_road_agents(
    pool: RoadAgentPool,
    spawn_points: List[RoadSpawnPoint],
    dt: float,
    rng: np.random.Generator
) -> int:
    """Spawn agents at road spawn points."""
    spawned = 0
    
    for sp in spawn_points:
        # Poisson-like spawning
        expected = sp.rate * dt
        count = int(expected)
        if rng.random() < (expected - count):
            count += 1
        
        for _ in range(count):
            # Small random offset along road
            offset = rng.uniform(-2, 2)
            dist = max(0, sp.distance_along + offset)
            
            agent_id = pool.spawn(
                road_id=sp.road_id,
                distance_along=dist,
                target_exit_id=sp.target_exit_id
            )
            
            if agent_id is not None:
                spawned += 1
    
    return spawned
