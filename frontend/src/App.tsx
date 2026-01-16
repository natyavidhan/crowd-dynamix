/**
 * Main application component.
 * Combines map view and dashboard.
 */

import { useEffect } from 'react';
import { MapView } from '@/components/map';
import { ControlPanel, ChokePointList, ChokePointCard } from '@/components/dashboard';
import { useSimulationStore } from '@/store';

export default function App() {
  const connect = useSimulationStore((state) => state.connect);
  const selectedChokePoint = useSimulationStore((state) => state.selectedChokePoint);
  
  // Connect to WebSocket on mount
  useEffect(() => {
    connect();
  }, [connect]);
  
  return (
    <div className="h-screen w-screen flex">
      {/* Map area */}
      <div className="flex-1 relative">
        <MapView />
        
        {/* Title overlay */}
        <div className="absolute top-4 left-4 bg-gray-900/90 backdrop-blur px-4 py-2 rounded-lg">
          <h1 className="text-xl font-bold">Crowd Instability Monitor</h1>
          <p className="text-sm text-gray-400">Pre-Stampede Early Warning System</p>
        </div>
        
        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-gray-900/90 backdrop-blur px-4 py-3 rounded-lg">
          <div className="text-xs text-gray-400 mb-2">Risk Level</div>
          <div className="flex gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-risk-green" />
              <span>Safe</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-risk-yellow" />
              <span>Unstable</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-risk-red" />
              <span>High Risk</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Dashboard sidebar */}
      <div className="w-96 bg-gray-900 border-l border-gray-800 p-4 overflow-y-auto flex flex-col gap-4">
        <ControlPanel />
        
        <div>
          <h2 className="text-lg font-bold mb-2">Choke Points</h2>
          <ChokePointList />
        </div>
        
        {/* Selected choke point details */}
        {selectedChokePoint && (
          <div>
            <h2 className="text-lg font-bold mb-2">Details</h2>
            <ChokePointCard chokePointId={selectedChokePoint} />
          </div>
        )}
      </div>
    </div>
  );
}
