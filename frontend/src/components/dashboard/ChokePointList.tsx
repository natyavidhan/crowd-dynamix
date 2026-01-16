/**
 * List of all choke points with summary view.
 */

import { useSimulationStore } from '@/store';
import { ciiToColor, riskLevelColor } from '@/utils';
import type { ChokePoint } from '@/types';

export function ChokePointList() {
  const chokePoints = useSimulationStore((state) => state.chokePoints);
  const selectedId = useSimulationStore((state) => state.selectedChokePoint);
  const selectChokePoint = useSimulationStore((state) => state.selectChokePoint);
  
  if (chokePoints.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 text-center text-gray-500">
        No choke points defined
      </div>
    );
  }
  
  return (
    <div className="space-y-2">
      {chokePoints.map((cp: ChokePoint) => (
        <button
          key={cp.id}
          onClick={() => selectChokePoint(selectedId === cp.id ? null : cp.id)}
          className={`w-full text-left p-3 rounded-lg transition ${
            selectedId === cp.id
              ? 'bg-gray-700 ring-2 ring-blue-500'
              : 'bg-gray-800 hover:bg-gray-750'
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="font-medium">{cp.name}</span>
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: riskLevelColor(cp.risk_state.level) }}
            />
          </div>
          <div className="flex items-center justify-between mt-1">
            <span
              className="text-lg font-bold font-mono"
              style={{ color: ciiToColor(cp.risk_state.cii) }}
            >
              {(cp.risk_state.cii * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-gray-500 capitalize">
              {cp.risk_state.trend === 'rising' && '↑ '}
              {cp.risk_state.trend === 'falling' && '↓ '}
              {cp.risk_state.level}
            </span>
          </div>
        </button>
      ))}
    </div>
  );
}

export default ChokePointList;
