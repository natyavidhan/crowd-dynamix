/**
 * Main map view component.
 * Displays OSM tiles, choke points, and agent overlay.
 */

import { useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, ZoomControl } from 'react-leaflet';
import { useSimulationStore } from '@/store';
import { ChokePointLayer } from './ChokePointLayer';
import { AgentCanvas } from './AgentCanvas';

// Default center (Bangalore - can be changed via API)
const DEFAULT_CENTER: [number, number] = [12.9716, 77.5946];
const DEFAULT_ZOOM = 17;

export function MapView() {
  const chokePoints = useSimulationStore((state) => state.chokePoints);
  const showAgents = useSimulationStore((state) => state.showAgents);
  const selectChokePoint = useSimulationStore((state) => state.selectChokePoint);
  
  // Calculate center from choke points
  const center = useMemo<[number, number]>(() => {
    if (chokePoints.length === 0) return DEFAULT_CENTER;
    
    const avgLat = chokePoints.reduce((sum, cp) => sum + cp.center.lat, 0) / chokePoints.length;
    const avgLng = chokePoints.reduce((sum, cp) => sum + cp.center.lng, 0) / chokePoints.length;
    
    return [avgLat, avgLng];
  }, [chokePoints]);
  
  return (
    <MapContainer
      center={center}
      zoom={DEFAULT_ZOOM}
      zoomControl={false}
      className="h-full w-full"
    >
      {/* OSM Tiles */}
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      
      {/* Zoom controls (bottom right) */}
      <ZoomControl position="bottomright" />
      
      {/* Choke point visualization */}
      <ChokePointLayer onSelect={selectChokePoint} />
      
      {/* Agent particles */}
      <AgentCanvas visible={showAgents} />
    </MapContainer>
  );
}

export default MapView;
