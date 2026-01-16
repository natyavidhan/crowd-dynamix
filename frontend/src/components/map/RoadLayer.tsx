/**
 * RoadLayer component for displaying venue roads on the map.
 */

import { Polyline, Tooltip } from 'react-leaflet';
import { useSimulationStore } from '@/store';
import type { LatLngExpression } from 'leaflet';

export function RoadLayer() {
  const currentVenue = useSimulationStore((state) => state.currentVenue);
  const showRoads = useSimulationStore((state) => state.showRoads);

  if (!showRoads || !currentVenue?.roads) {
    return null;
  }

  return (
    <>
      {currentVenue.roads.map((road) => {
        const positions: LatLngExpression[] = road.coordinates.map((coord) => [
          coord.lat,
          coord.lng,
        ]);

        // Calculate visual width based on road width (scaled)
        // Typical road widths: 3m (narrow lane) to 20m (main road)
        const visualWeight = Math.max(2, Math.min(12, road.width / 2));

        return (
          <Polyline
            key={road.id}
            positions={positions}
            pathOptions={{
              color: '#60a5fa', // blue-400
              weight: visualWeight,
              opacity: 0.5,
              lineCap: 'round',
              lineJoin: 'round',
            }}
          >
            <Tooltip permanent={false} direction="top" offset={[0, -10]}>
              <div className="text-xs">
                <div className="font-medium">{road.name}</div>
                <div className="text-gray-500">Width: {road.width}m</div>
              </div>
            </Tooltip>
          </Polyline>
        );
      })}
    </>
  );
}
