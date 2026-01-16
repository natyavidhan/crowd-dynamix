/**
 * Choke point visualization layer.
 * Renders circular/polygonal regions with risk-based coloring.
 */

import { Circle, Popup } from 'react-leaflet';
import { useSimulationStore } from '@/store';
import { ciiToColor } from '@/utils';
import type { ChokePoint } from '@/types';

interface ChokePointLayerProps {
  onSelect?: (id: string) => void;
}

export function ChokePointLayer({ onSelect }: ChokePointLayerProps) {
  const chokePoints = useSimulationStore((state) => state.chokePoints);
  const selectedId = useSimulationStore((state) => state.selectedChokePoint);
  
  return (
    <>
      {chokePoints.map((cp) => (
        <ChokePointCircle
          key={cp.id}
          chokePoint={cp}
          selected={cp.id === selectedId}
          onSelect={onSelect}
        />
      ))}
    </>
  );
}

interface ChokePointCircleProps {
  chokePoint: ChokePoint;
  selected: boolean;
  onSelect?: (id: string) => void;
}

function ChokePointCircle({ chokePoint, selected, onSelect }: ChokePointCircleProps) {
  const { center, geometry, risk_state, name, id } = chokePoint;
  
  // Only support circle geometry for now
  const radius = geometry.type === 'circle' ? geometry.radius : 15;
  
  // Color based on CII
  const color = ciiToColor(risk_state.cii);
  const opacity = 0.3 + risk_state.cii * 0.4;
  
  return (
    <Circle
      center={[center.lat, center.lng]}
      radius={radius}
      pathOptions={{
        color: selected ? '#ffffff' : color,
        fillColor: color,
        fillOpacity: opacity,
        weight: selected ? 3 : 2,
      }}
      eventHandlers={{
        click: () => onSelect?.(id),
      }}
    >
      <Popup>
        <div className="text-gray-900">
          <h3 className="font-bold text-lg">{name}</h3>
          <p className="text-sm">
            CII: <span style={{ color }}>{(risk_state.cii * 100).toFixed(1)}%</span>
          </p>
          <p className="text-sm">
            Status: <span className="capitalize">{risk_state.level}</span>
          </p>
          <p className="text-sm">
            Trend: <span className="capitalize">{risk_state.trend}</span>
          </p>
        </div>
      </Popup>
    </Circle>
  );
}

export default ChokePointLayer;
