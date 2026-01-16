/**
 * Type definitions for the simulation state.
 * Mirrors the Python backend models.
 */

// ============================================================================
// Geographic Types
// ============================================================================

export interface GeoPoint {
  lat: number;
  lng: number;
}

export interface SimPoint {
  x: number;
  y: number;
}

// ============================================================================
// Choke Point
// ============================================================================

export type SensorType = 'mmwave' | 'audio' | 'camera';

export interface SensorConfig {
  type: SensorType;
  enabled: boolean;
  confidence: number;
}

export interface CircleGeometry {
  type: 'circle';
  radius: number;
}

export interface PolygonGeometry {
  type: 'polygon';
  vertices: GeoPoint[];
}

export type ChokePointGeometry = CircleGeometry | PolygonGeometry;

export interface RiskState {
  cii: number;
  trend: 'stable' | 'rising' | 'falling';
  level: 'green' | 'yellow' | 'red';
  last_updated: number;
}

export interface ChokePoint {
  id: string;
  name: string;
  center: GeoPoint;
  geometry: ChokePointGeometry;
  sensors: SensorConfig[];
  risk_state: RiskState;
  sim_center?: SimPoint;
  sim_radius?: number;
}

// ============================================================================
// Agents
// ============================================================================

export type AgentState = 'moving' | 'slowing' | 'stopped' | 'pushing';

export interface AgentSnapshot {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  state: AgentState;
}

// ============================================================================
// Sensor Data
// ============================================================================

export interface MmWaveSensorData {
  timestamp: number;
  choke_point_id: string;
  avg_velocity: number;
  velocity_variance: number;
  dominant_direction: number;
  directional_divergence: number;
  stop_go_frequency: number;
  stationary_ratio: number;
}

export interface AudioSensorData {
  timestamp: number;
  choke_point_id: string;
  sound_energy_level: number;
  spike_detected: boolean;
  spike_intensity: number;
  audio_character: 'ambient' | 'loud' | 'distressed';
}

export interface CameraSensorData {
  timestamp: number;
  choke_point_id: string;
  crowd_density: number;
  occlusion_percentage: number;
  optical_flow_consistency: number;
}

export interface AggregatedSensorData {
  choke_point_id: string;
  timestamp: number;
  mmwave: MmWaveSensorData | null;
  audio: AudioSensorData | null;
  camera: CameraSensorData | null;
  overall_confidence: number;
}

// ============================================================================
// CII
// ============================================================================

export interface CIIContribution {
  factor: string;
  raw_value: number;
  normalized_value: number;
  weight: number;
  contribution: number;
}

export interface CIIExplanation {
  cii: number;
  contributions: CIIContribution[];
  audio_modifier: number;
  interpretation: string;
}

// ============================================================================
// Simulation Config & State
// ============================================================================

export interface SimulationConfig {
  running: boolean;
  speed_multiplier: number;
  base_inflow_rate: number;
  current_inflow_multiplier: number;
  max_agents: number;
  opposing_flow_enabled: boolean;
  blocked_exits: string[];
  density_surge_enabled: boolean;
}

export interface SimulationState {
  timestamp: number;
  tick: number;
  agents: AgentSnapshot[];
  choke_points: ChokePoint[];
  sensor_data: Record<string, AggregatedSensorData>;
  cii_explanations: Record<string, CIIExplanation>;
  config: SimulationConfig;
}

// ============================================================================
// Control Messages
// ============================================================================

export type ControlAction =
  | 'start'
  | 'stop'
  | 'reset'
  | 'set_speed'
  | 'set_inflow'
  | 'toggle_opposing_flow'
  | 'toggle_density_surge'
  | 'block_exit'
  | 'unblock_exit'
  | 'add_choke_point'
  | 'remove_choke_point';

export interface ControlMessage {
  action: ControlAction;
  payload?: Record<string, unknown>;
}
