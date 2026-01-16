/**
 * Choke point detail card.
 * Shows CII, sensor data, and explanations for selected choke point.
 */

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { useSimulationStore } from '@/store';
import { ciiToColor, riskLevelColor } from '@/utils';
import type { ChokePoint, CIIExplanation, AggregatedSensorData } from '@/types';

interface ChokePointCardProps {
  chokePointId: string;
}

export function ChokePointCard({ chokePointId }: ChokePointCardProps) {
  const chokePoints = useSimulationStore((state) => state.chokePoints);
  const sensorData = useSimulationStore((state) => state.sensorData);
  const ciiExplanations = useSimulationStore((state) => state.ciiExplanations);
  const ciiHistory = useSimulationStore((state) => state.ciiHistory);
  
  const chokePoint = chokePoints.find((cp: ChokePoint) => cp.id === chokePointId);
  const sensors = sensorData[chokePointId];
  const explanation = ciiExplanations[chokePointId];
  const history = ciiHistory[chokePointId] || [];
  
  if (!chokePoint) return null;
  
  const { risk_state, name, sensors: sensorConfigs } = chokePoint;
  const color = ciiToColor(risk_state.cii);
  const levelColor = riskLevelColor(risk_state.level);
  
  // Format history for chart
  const chartData = useMemo(() => {
    const now = Date.now();
    return history.slice(-60).map((d: { timestamp: number; cii: number }) => ({
      time: Math.round((d.timestamp - now) / 1000),
      cii: d.cii,
    }));
  }, [history]);
  
  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-bold">{name}</h3>
        <div
          className="px-3 py-1 rounded-full text-sm font-medium"
          style={{ backgroundColor: levelColor + '30', color: levelColor }}
        >
          {risk_state.level.toUpperCase()}
        </div>
      </div>
      
      {/* CII Gauge */}
      <div className="flex items-center gap-4">
        <div className="text-4xl font-bold" style={{ color }}>
          {(risk_state.cii * 100).toFixed(1)}%
        </div>
        <div className="flex-1">
          <div className="text-sm text-gray-400">Crowd Instability Index</div>
          <div className="flex items-center gap-1 text-sm">
            <span
              className={
                risk_state.trend === 'rising'
                  ? 'text-red-400'
                  : risk_state.trend === 'falling'
                  ? 'text-green-400'
                  : 'text-gray-400'
              }
            >
              {risk_state.trend === 'rising'
                ? '↑'
                : risk_state.trend === 'falling'
                ? '↓'
                : '→'}
            </span>
            <span className="text-gray-400 capitalize">{risk_state.trend}</span>
          </div>
        </div>
      </div>
      
      {/* CII Chart */}
      <div className="h-24">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <XAxis
              dataKey="time"
              tick={{ fontSize: 10, fill: '#9ca3af' }}
              tickFormatter={(v) => `${v}s`}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: '#9ca3af' }}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              width={35}
            />
            <ReferenceLine y={0.35} stroke="#eab308" strokeDasharray="3 3" />
            <ReferenceLine y={0.65} stroke="#dc2626" strokeDasharray="3 3" />
            <Line
              type="monotone"
              dataKey="cii"
              stroke={color}
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      {/* Sensor status */}
      <div>
        <div className="text-sm text-gray-400 mb-2">Sensors</div>
        <div className="flex gap-2">
          {sensorConfigs.map((s) => (
            <div
              key={s.type}
              className={`px-2 py-1 rounded text-xs ${
                s.enabled ? 'bg-green-900 text-green-300' : 'bg-gray-700 text-gray-500'
              }`}
            >
              {s.type.toUpperCase()}
              {s.enabled && sensors?.overall_confidence && (
                <span className="ml-1 opacity-70">
                  {(s.confidence * 100).toFixed(0)}%
                </span>
              )}
            </div>
          ))}
        </div>
      </div>
      
      {/* CII Breakdown */}
      {explanation && (
        <div>
          <div className="text-sm text-gray-400 mb-2">Risk Factors</div>
          <div className="space-y-2">
            {explanation.contributions.map((c) => (
              <div key={c.factor} className="flex items-center gap-2">
                <div className="flex-1 text-sm">{c.factor}</div>
                <div className="w-24 bg-gray-700 rounded-full h-2">
                  <div
                    className="h-2 rounded-full transition-all duration-300"
                    style={{
                      width: `${c.normalized_value * 100}%`,
                      backgroundColor: ciiToColor(c.normalized_value),
                    }}
                  />
                </div>
                <div className="w-12 text-right text-sm font-mono">
                  {(c.contribution * 100).toFixed(1)}%
                </div>
              </div>
            ))}
            
            {explanation.audio_modifier > 1.01 && (
              <div className="text-sm text-orange-400">
                Audio modifier: ×{explanation.audio_modifier.toFixed(2)}
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Interpretation */}
      {explanation?.interpretation && (
        <div className="text-sm text-gray-300 italic border-l-2 border-gray-600 pl-3">
          {explanation.interpretation}
        </div>
      )}
      
      {/* Raw sensor data (collapsed) */}
      {sensors?.mmwave && (
        <details className="text-xs text-gray-500">
          <summary className="cursor-pointer hover:text-gray-400">
            Raw Sensor Data
          </summary>
          <div className="mt-2 space-y-1 font-mono">
            <div>Avg Velocity: {sensors.mmwave.avg_velocity.toFixed(2)} m/s</div>
            <div>Velocity Var: {sensors.mmwave.velocity_variance.toFixed(3)} m²/s²</div>
            <div>Stop-Go Freq: {sensors.mmwave.stop_go_frequency.toFixed(1)} /min</div>
            <div>Dir. Divergence: {(sensors.mmwave.directional_divergence * 100).toFixed(1)}%</div>
            {sensors.camera && (
              <div>Density: {sensors.camera.crowd_density.toFixed(2)} p/m²</div>
            )}
          </div>
        </details>
      )}
    </div>
  );
}

export default ChokePointCard;
