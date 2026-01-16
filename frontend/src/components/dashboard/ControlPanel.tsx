/**
 * Simulation control panel.
 * Start/stop, speed, inflow, perturbations.
 */

import { useSimulationStore } from '@/store';

export function ControlPanel() {
  const config = useSimulationStore((state) => state.config);
  const connected = useSimulationStore((state) => state.connected);
  const sendControl = useSimulationStore((state) => state.sendControl);
  const agents = useSimulationStore((state) => state.agents);
  const showAgents = useSimulationStore((state) => state.showAgents);
  const toggleAgents = useSimulationStore((state) => state.toggleAgents);
  
  const handleStart = () => sendControl({ action: 'start' });
  const handleStop = () => sendControl({ action: 'stop' });
  const handleReset = () => sendControl({ action: 'reset' });
  
  const handleSpeedChange = (value: number) => {
    sendControl({ action: 'set_speed', payload: { multiplier: value } });
  };
  
  const handleInflowChange = (value: number) => {
    sendControl({ action: 'set_inflow', payload: { multiplier: value } });
  };
  
  const handleOpposingFlow = () => {
    sendControl({
      action: 'toggle_opposing_flow',
      payload: { enabled: !config.opposing_flow_enabled },
    });
  };
  
  const handleDensitySurge = () => {
    sendControl({
      action: 'toggle_density_surge',
      payload: { enabled: !config.density_surge_enabled },
    });
  };
  
  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold">Simulation Controls</h2>
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              connected ? 'bg-green-500' : 'bg-red-500'
            }`}
          />
          <span className="text-sm text-gray-400">
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>
      
      {/* Playback controls */}
      <div className="flex gap-2">
        <button
          onClick={config.running ? handleStop : handleStart}
          className={`flex-1 px-4 py-2 rounded font-medium transition ${
            config.running
              ? 'bg-yellow-600 hover:bg-yellow-700'
              : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          {config.running ? '⏸ Pause' : '▶ Start'}
        </button>
        <button
          onClick={handleReset}
          className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition"
        >
          ↺ Reset
        </button>
      </div>
      
      {/* Stats */}
      <div className="text-sm text-gray-400">
        Agents: <span className="text-white font-mono">{agents.length}</span>
      </div>
      
      {/* Speed control */}
      <div>
        <label className="block text-sm text-gray-400 mb-1">
          Speed: {config.speed_multiplier.toFixed(1)}x
        </label>
        <input
          type="range"
          min="0.1"
          max="5"
          step="0.1"
          value={config.speed_multiplier}
          onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
          className="w-full accent-blue-500"
        />
      </div>
      
      {/* Inflow control */}
      <div>
        <label className="block text-sm text-gray-400 mb-1">
          Crowd Inflow: {(config.current_inflow_multiplier * 100).toFixed(0)}%
        </label>
        <input
          type="range"
          min="0"
          max="3"
          step="0.1"
          value={config.current_inflow_multiplier}
          onChange={(e) => handleInflowChange(parseFloat(e.target.value))}
          className="w-full accent-blue-500"
        />
      </div>
      
      {/* Perturbations */}
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-gray-300">Perturbations</h3>
        
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={config.opposing_flow_enabled}
            onChange={handleOpposingFlow}
            className="rounded accent-orange-500"
          />
          <span className="text-sm">Opposing Flow</span>
        </label>
        
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={config.density_surge_enabled}
            onChange={handleDensitySurge}
            className="rounded accent-red-500"
          />
          <span className="text-sm">Density Surge</span>
        </label>
      </div>
      
      {/* Agent visibility */}
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={showAgents}
          onChange={toggleAgents}
          className="rounded accent-blue-500"
        />
        <span className="text-sm">Show Agents</span>
      </label>
    </div>
  );
}

export default ControlPanel;
