/**
 * Zustand store for simulation state management.
 */

import { create } from 'zustand';
import type {
  SimulationState,
  SimulationConfig,
  ChokePoint,
  AgentSnapshot,
  AggregatedSensorData,
  CIIExplanation,
  ControlMessage,
  VenueSummary,
  VenueInfo,
} from '@/types';

// ============================================================================
// Store State
// ============================================================================

interface SimulationStore {
  // Connection state
  connected: boolean;
  ws: WebSocket | null;
  
  // Simulation state (from backend)
  tick: number;
  agents: AgentSnapshot[];
  chokePoints: ChokePoint[];
  sensorData: Record<string, AggregatedSensorData>;
  ciiExplanations: Record<string, CIIExplanation>;
  config: SimulationConfig;
  
  // Venue state
  availableVenues: VenueSummary[];
  currentVenue: VenueInfo | null;
  venueLoading: boolean;
  
  // UI state
  selectedChokePoint: string | null;
  showAgents: boolean;
  showRoads: boolean;
  
  // Time series history (for charts)
  ciiHistory: Record<string, { timestamp: number; cii: number }[]>;
  lastHistoryUpdate: number | null;
  
  // Actions
  connect: () => void;
  disconnect: () => void;
  sendControl: (message: ControlMessage) => void;
  updateState: (state: SimulationState) => void;
  selectChokePoint: (id: string | null) => void;
  toggleAgents: () => void;
  toggleRoads: () => void;
  fetchVenues: () => Promise<void>;
  loadVenue: (venueId: string) => Promise<void>;
  fetchCurrentVenue: () => Promise<void>;
}

// ============================================================================
// Default Config
// ============================================================================

const defaultConfig: SimulationConfig = {
  running: false,
  speed_multiplier: 1.0,
  base_inflow_rate: 5.0,
  current_inflow_multiplier: 1.0,
  max_agents: 500,
  opposing_flow_enabled: false,
  blocked_exits: [],
  density_surge_enabled: false,
};

// ============================================================================
// WebSocket URL
// ============================================================================

const getWebSocketUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname;
  // In development, backend is on port 8000
  const port = import.meta.env.DEV ? '8000' : window.location.port;
  return `${protocol}//${host}:${port}/ws`;
};

const getApiBaseUrl = () => {
  const host = window.location.hostname;
  const port = import.meta.env.DEV ? '8000' : window.location.port;
  return `http://${host}:${port}`;
};

// ============================================================================
// Store
// ============================================================================

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  // Initial state
  connected: false,
  ws: null,
  tick: 0,
  agents: [],
  chokePoints: [],
  sensorData: {},
  ciiExplanations: {},
  config: defaultConfig,
  selectedChokePoint: null,
  showAgents: true,
  showRoads: true,
  ciiHistory: {},
  lastHistoryUpdate: null,
  
  // Venue state
  availableVenues: [],
  currentVenue: null,
  venueLoading: false,
  
  // Connect to WebSocket
  connect: () => {
    const ws = new WebSocket(getWebSocketUrl());
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      set({ connected: true, ws });
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      set({ connected: false, ws: null });
      
      // Reconnect after 2 seconds
      setTimeout(() => {
        if (!get().connected) {
          get().connect();
        }
      }, 2000);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onmessage = (event) => {
      try {
        const state: SimulationState = JSON.parse(event.data);
        get().updateState(state);
      } catch (error) {
        console.error('Failed to parse message:', error);
      }
    };
    
    set({ ws });
  },
  
  // Disconnect
  disconnect: () => {
    const { ws } = get();
    if (ws) {
      ws.close();
    }
    set({ connected: false, ws: null });
  },
  
  // Send control message
  sendControl: (message: ControlMessage) => {
    const { ws, connected } = get();
    if (ws && connected) {
      ws.send(JSON.stringify(message));
    }
  },
  
  // Update state from backend
  updateState: (state: SimulationState) => {
    const { ciiHistory, lastHistoryUpdate } = get();
    
    // Update CII history for charts - throttle to every 500ms instead of every frame
    const now = Date.now();
    let newHistory = ciiHistory;
    
    if (!lastHistoryUpdate || now - lastHistoryUpdate > 500) {
      newHistory = { ...ciiHistory };
      
      for (const cp of state.choke_points) {
        if (!newHistory[cp.id]) {
          newHistory[cp.id] = [];
        }
        
        // Add new data point
        newHistory[cp.id] = [
          ...newHistory[cp.id].slice(-120), // Keep last 60 data points (at 500ms = 30 seconds)
          { timestamp: now, cii: cp.risk_state.cii }
        ];
      }
    }
    
    set({
      tick: state.tick,
      agents: state.agents,
      chokePoints: state.choke_points,
      sensorData: state.sensor_data,
      ciiExplanations: state.cii_explanations,
      config: state.config,
      ciiHistory: newHistory,
      lastHistoryUpdate: newHistory !== ciiHistory ? now : lastHistoryUpdate,
    });
  },
  
  // Select choke point
  selectChokePoint: (id: string | null) => {
    set({ selectedChokePoint: id });
  },
  
  // Toggle agent visibility
  toggleAgents: () => {
    set(state => ({ showAgents: !state.showAgents }));
  },
  
  // Toggle road visibility
  toggleRoads: () => {
    set(state => ({ showRoads: !state.showRoads }));
  },
  
  // Fetch available venues
  fetchVenues: async () => {
    try {
      const response = await fetch(`${getApiBaseUrl()}/venues`);
      const data = await response.json();
      set({ availableVenues: data.venues || [] });
    } catch (error) {
      console.error('Failed to fetch venues:', error);
    }
  },
  
  // Load a venue
  loadVenue: async (venueId: string) => {
    set({ venueLoading: true });
    try {
      const response = await fetch(`${getApiBaseUrl()}/venues/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ venue_id: venueId }),
      });
      
      if (response.ok) {
        // Fetch current venue to get road data
        await get().fetchCurrentVenue();
        // Clear CII history on venue change
        set({ ciiHistory: {} });
      }
    } catch (error) {
      console.error('Failed to load venue:', error);
    } finally {
      set({ venueLoading: false });
    }
  },
  
  // Fetch current venue info
  fetchCurrentVenue: async () => {
    try {
      const response = await fetch(`${getApiBaseUrl()}/venues/current`);
      const data = await response.json();
      set({ currentVenue: data.venue || null });
    } catch (error) {
      console.error('Failed to fetch current venue:', error);
    }
  },
}));
