/**
 * Main application component.
 * Combines map view and dashboard.
 */

import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { MapView } from '@/components/map';
import { ControlPanel, ChokePointList, ChokePointCard, VenueSelector } from '@/components/dashboard';
import { useSimulationStore } from '@/store';

export default function App() {
  const connect = useSimulationStore((state) => state.connect);
  const selectedChokePoint = useSimulationStore((state) => state.selectedChokePoint);
  const toggleRoads = useSimulationStore((state) => state.toggleRoads);
  const showRoads = useSimulationStore((state) => state.showRoads);
  const currentVenue = useSimulationStore((state) => state.currentVenue);
  
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
        <div className="absolute top-4 left-4 bg-gray-900/90 backdrop-blur px-4 py-2 rounded-lg z-[1000] pointer-events-auto">
          <h1 className="text-xl font-bold">Crowd Instability Monitor</h1>
          <p className="text-sm text-gray-400">Pre-Stampede Early Warning System</p>
          {currentVenue && (
            <p className="text-xs text-blue-400 mt-1">{currentVenue.name}</p>
          )}
        </div>
        
        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-gray-900/90 backdrop-blur px-4 py-3 rounded-lg z-[1000] pointer-events-auto">
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
          
          {/* Road toggle */}
          <div className="mt-2 pt-2 border-t border-gray-700 flex items-center gap-2">
            <input
              type="checkbox"
              id="showRoads"
              checked={showRoads}
              onChange={toggleRoads}
              className="rounded border-gray-600"
            />
            <label htmlFor="showRoads" className="text-xs text-gray-400">Show roads</label>
          </div>
          
          {/* Editor link */}
          <div className="mt-2 pt-2 border-t border-gray-700">
            <Link
              to="/editor"
              className="text-xs text-purple-400 hover:text-purple-300 underline"
            >
              🗺️ Open Venue Editor
            </Link>
          </div>
        </div>
      </div>
      
      {/* Dashboard sidebar */}
      <div className="w-96 bg-gray-900 border-l border-gray-800 p-4 overflow-y-auto flex flex-col gap-4">
        {/* Venue selector */}
        <VenueSelector />
        
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
