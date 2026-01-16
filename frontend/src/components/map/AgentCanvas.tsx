/**
 * Agent canvas overlay for rendering crowd particles.
 * Uses HTML5 Canvas for high performance with 500+ agents.
 */

import { useEffect, useRef, useMemo } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';
import { useSimulationStore } from '@/store';
import { agentStateColor, createCoordinateSystem, simToLatLng } from '@/utils';
import type { AgentSnapshot } from '@/types';

// Default map origin (Bangalore)
const DEFAULT_ORIGIN = { lat: 12.9716, lng: 77.5946 };

interface AgentCanvasProps {
  visible?: boolean;
}

export function AgentCanvas({ visible = true }: AgentCanvasProps) {
  const map = useMap();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const agents = useSimulationStore((state) => state.agents);
  const chokePoints = useSimulationStore((state) => state.chokePoints);
  const origin = useSimulationStore((state) => state.origin);
  const currentVenue = useSimulationStore((state) => state.currentVenue);
  
  // Get origin from WebSocket state (most reliable), then venue, then choke points, then default
  const effectiveOrigin = useMemo(() => {
    // Prefer origin from WebSocket state (always up-to-date)
    if (origin) {
      return origin;
    }
    // Fallback to venue origin from REST API
    if (currentVenue?.origin) {
      return currentVenue.origin;
    }
    // Fallback to first choke point center
    if (chokePoints.length > 0) {
      return chokePoints[0].center;
    }
    return DEFAULT_ORIGIN;
  }, [origin, currentVenue, chokePoints]);
  
  const coordSystem = useMemo(() => createCoordinateSystem(effectiveOrigin), [effectiveOrigin]);
  
  // Create canvas pane
  useEffect(() => {
    const pane = map.createPane('agentPane');
    pane.style.zIndex = '450';
    
    const canvas = L.DomUtil.create('canvas', 'agent-canvas', pane);
    canvas.style.position = 'absolute';
    canvas.style.pointerEvents = 'none';
    canvasRef.current = canvas;
    
    const updateCanvasPosition = () => {
      const size = map.getSize();
      const topLeft = map.containerPointToLayerPoint([0, 0]);
      
      canvas.width = size.x;
      canvas.height = size.y;
      L.DomUtil.setPosition(canvas, topLeft);
    };
    
    updateCanvasPosition();
    map.on('move zoom resize', updateCanvasPosition);
    
    return () => {
      map.off('move zoom resize', updateCanvasPosition);
      if (canvas.parentNode) {
        canvas.parentNode.removeChild(canvas);
      }
    };
  }, [map]);
  
  // Render agents
  useEffect(() => {
    if (!canvasRef.current || !visible) {
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
      return;
    }
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Render each agent
    for (const agent of agents) {
      renderAgent(ctx, agent, map, coordSystem);
    }
  }, [agents, map, visible, coordSystem]);
  
  return null;
}

function renderAgent(
  ctx: CanvasRenderingContext2D,
  agent: AgentSnapshot,
  map: L.Map,
  coordSystem: ReturnType<typeof createCoordinateSystem>
) {
  // Convert sim coords to lat/lng
  const latLng = simToLatLng({ x: agent.x, y: agent.y }, coordSystem);
  
  // Convert to screen coords
  const point = map.latLngToContainerPoint(latLng);
  
  // Skip if off-screen
  const size = map.getSize();
  if (point.x < -10 || point.x > size.x + 10 || point.y < -10 || point.y > size.y + 10) {
    return;
  }
  
  // Draw agent circle
  const color = agentStateColor(agent.state);
  const radius = 4;
  
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  
  // Add slight border for visibility
  ctx.strokeStyle = 'rgba(0,0,0,0.3)';
  ctx.lineWidth = 1;
  ctx.stroke();
}

export default AgentCanvas;
