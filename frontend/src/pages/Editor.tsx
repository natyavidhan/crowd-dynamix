/**
 * Map-based Venue YAML Editor
 * Visual GUI for creating venue configurations with roads, spawn points, choke points, etc.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import * as yaml from 'js-yaml';

// ============================================================================
// Types
// ============================================================================

interface GeoPoint {
  lat: number;
  lng: number;
}

interface Road {
  id: string;
  name: string;
  width: number;
  points: GeoPoint[];
}

interface SpawnPoint {
  id: string;
  name: string;
  lat: number;
  lng: number;
  rate: number;
}

interface ChokePoint {
  id: string;
  name: string;
  lat: number;
  lng: number;
  radius: number;
}

interface Exit {
  id: string;
  name: string;
  lat: number;
  lng: number;
}

interface VenueData {
  venue: {
    name: string;
    location: string;
    description: string;
  };
  origin: GeoPoint;
  roads: Road[];
  spawns: SpawnPoint[];
  chokePoints: ChokePoint[];
  exits: Exit[];
}

type EditorMode = 'pan' | 'road' | 'spawn' | 'choke' | 'exit' | 'origin' | 'erase';

// ============================================================================
// Utility Functions
// ============================================================================

function uid(prefix: string = ''): string {
  return prefix + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function createEmptyVenue(): VenueData {
  return {
    venue: {
      name: 'New Venue',
      location: '',
      description: '',
    },
    origin: { lat: 13.05, lng: 80.28 },
    roads: [],
    spawns: [],
    chokePoints: [],
    exits: [],
  };
}

// ============================================================================
// YAML Export/Import
// ============================================================================

function exportToYAML(data: VenueData): string {
  // Find first exit (default destination for spawn points)
  const defaultExitId = data.exits.length > 0 ? data.exits[0].id : null;
  
  const yamlData = {
    venue: data.venue,
    origin: data.origin,
    roads: data.roads.map(r => ({
      id: r.id,
      name: r.name,
      width: r.width,
      speed_limit: 1.5,
      bidirectional: true,
      centerline: r.points.map(p => ({ lat: p.lat, lng: p.lng })),
    })),
    exits: data.exits.map(e => ({
      id: e.id,
      name: e.name,
      location: { lat: e.lat, lng: e.lng },
      capacity: 100,
    })),
    spawn_points: data.spawns.map(s => ({
      id: s.id,
      name: s.name,
      location: { lat: s.lat, lng: s.lng },
      rate: s.rate,
      spread: 3.0,
      flow_group: 'default',
      // Link spawn to exit so agents have a destination
      ...(defaultExitId ? { exit_id: defaultExitId } : {}),
    })),
    choke_points: data.chokePoints.map(c => ({
      id: c.id,
      name: c.name,
      location: { lat: c.lat, lng: c.lng },
      radius: c.radius,
      mmwave_enabled: true,
      audio_enabled: true,
      camera_enabled: true,
    })),
    defaults: {
      agent_count: 300,
      spawn_rate: 8.0,
      speed_multiplier: 1.0,
    },
  };

  return yaml.dump(yamlData, { lineWidth: 120, noRefs: true });
}

function importFromYAML(yamlText: string): VenueData | null {
  try {
    const parsed = yaml.load(yamlText) as any;
    
    const data: VenueData = createEmptyVenue();
    
    if (parsed.venue) {
      data.venue = {
        name: parsed.venue.name || 'Imported Venue',
        location: parsed.venue.location || '',
        description: parsed.venue.description || '',
      };
    }
    
    if (parsed.origin) {
      data.origin = { lat: parsed.origin.lat, lng: parsed.origin.lng };
    }
    
    if (parsed.roads && Array.isArray(parsed.roads)) {
      data.roads = parsed.roads.map((r: any) => ({
        id: r.id || uid('road_'),
        name: r.name || '',
        width: r.width || 10,
        points: (r.centerline || []).map((p: any) => ({ lat: p.lat, lng: p.lng })),
      }));
    }
    
    if (parsed.spawn_points && Array.isArray(parsed.spawn_points)) {
      data.spawns = parsed.spawn_points.map((s: any) => ({
        id: s.id || uid('spawn_'),
        name: s.name || '',
        lat: s.location?.lat || s.lat,
        lng: s.location?.lng || s.lng,
        rate: s.rate || 5.0,
      }));
    }
    
    if (parsed.choke_points && Array.isArray(parsed.choke_points)) {
      data.chokePoints = parsed.choke_points.map((c: any) => ({
        id: c.id || uid('choke_'),
        name: c.name || '',
        lat: c.location?.lat || c.lat,
        lng: c.location?.lng || c.lng,
        radius: c.radius || 20,
      }));
    }
    
    if (parsed.exits && Array.isArray(parsed.exits)) {
      data.exits = parsed.exits.map((e: any) => ({
        id: e.id || uid('exit_'),
        name: e.name || '',
        lat: e.location?.lat || e.lat,
        lng: e.location?.lng || e.lng,
      }));
    }
    
    return data;
  } catch (err) {
    console.error('YAML parse error:', err);
    return null;
  }
}

// ============================================================================
// Editor Component
// ============================================================================

export default function Editor() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layersRef = useRef<{
    roads: Record<string, L.Polyline>;
    spawns: Record<string, L.Marker>;
    chokePoints: Record<string, L.Circle>;
    exits: Record<string, L.CircleMarker>;
    origin: L.Marker | null;
    preview: L.Polyline | null;
  }>({
    roads: {},
    spawns: {},
    chokePoints: {},
    exits: {},
    origin: null,
    preview: null,
  });

  const [data, setData] = useState<VenueData>(createEmptyVenue);
  const [mode, setMode] = useState<EditorMode>('pan');
  const [currentRoad, setCurrentRoad] = useState<Road | null>(null);
  const [showYamlPanel, setShowYamlPanel] = useState(false);
  const [yamlText, setYamlText] = useState('');
  // Reserved for future use: selecting items in sidebar
  const [_selectedItem, _setSelectedItem] = useState<{ type: string; id: string } | null>(null);

  // Initialize map
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = L.map(mapContainerRef.current).setView([data.origin.lat, data.origin.lng], 16);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map);

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Handle map clicks
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const handleClick = (e: L.LeafletMouseEvent) => {
      const { lat, lng } = e.latlng;

      if (mode === 'road' && currentRoad) {
        setCurrentRoad(prev => {
          if (!prev) return prev;
          return { ...prev, points: [...prev.points, { lat, lng }] };
        });
      } else if (mode === 'spawn') {
        const newSpawn: SpawnPoint = {
          id: uid('spawn_'),
          name: `Spawn ${data.spawns.length + 1}`,
          lat,
          lng,
          rate: 5.0,
        };
        setData(prev => ({ ...prev, spawns: [...prev.spawns, newSpawn] }));
      } else if (mode === 'choke') {
        const newChoke: ChokePoint = {
          id: uid('choke_'),
          name: `Choke Point ${data.chokePoints.length + 1}`,
          lat,
          lng,
          radius: 25,
        };
        setData(prev => ({ ...prev, chokePoints: [...prev.chokePoints, newChoke] }));
      } else if (mode === 'exit') {
        const newExit: Exit = {
          id: uid('exit_'),
          name: `Exit ${data.exits.length + 1}`,
          lat,
          lng,
        };
        setData(prev => ({ ...prev, exits: [...prev.exits, newExit] }));
      } else if (mode === 'origin') {
        setData(prev => ({ ...prev, origin: { lat, lng } }));
      } else if (mode === 'erase') {
        eraseNearestFeature(lat, lng);
      }
    };

    map.on('click', handleClick);
    return () => {
      map.off('click', handleClick);
    };
  }, [mode, currentRoad, data]);

  // Erase nearest feature
  const eraseNearestFeature = useCallback((lat: number, lng: number) => {
    const map = mapRef.current;
    if (!map) return;

    const clickLatLng = L.latLng(lat, lng);
    const threshold = 30; // meters

    // Check spawns
    for (const spawn of data.spawns) {
      if (map.distance(clickLatLng, L.latLng(spawn.lat, spawn.lng)) < threshold) {
        setData(prev => ({ ...prev, spawns: prev.spawns.filter(s => s.id !== spawn.id) }));
        return;
      }
    }

    // Check choke points
    for (const cp of data.chokePoints) {
      if (map.distance(clickLatLng, L.latLng(cp.lat, cp.lng)) < threshold) {
        setData(prev => ({ ...prev, chokePoints: prev.chokePoints.filter(c => c.id !== cp.id) }));
        return;
      }
    }

    // Check exits
    for (const exit of data.exits) {
      if (map.distance(clickLatLng, L.latLng(exit.lat, exit.lng)) < threshold) {
        setData(prev => ({ ...prev, exits: prev.exits.filter(e => e.id !== exit.id) }));
        return;
      }
    }

    // Check roads (any vertex)
    for (const road of data.roads) {
      for (const pt of road.points) {
        if (map.distance(clickLatLng, L.latLng(pt.lat, pt.lng)) < threshold) {
          setData(prev => ({ ...prev, roads: prev.roads.filter(r => r.id !== road.id) }));
          return;
        }
      }
    }
  }, [data]);

  // Render map layers
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const layers = layersRef.current;

    // Clear existing layers
    Object.values(layers.roads).forEach(l => map.removeLayer(l));
    Object.values(layers.spawns).forEach(l => map.removeLayer(l));
    Object.values(layers.chokePoints).forEach(l => map.removeLayer(l));
    Object.values(layers.exits).forEach(l => map.removeLayer(l));
    if (layers.origin) map.removeLayer(layers.origin);
    if (layers.preview) map.removeLayer(layers.preview);

    layers.roads = {};
    layers.spawns = {};
    layers.chokePoints = {};
    layers.exits = {};

    // Render roads
    data.roads.forEach(road => {
      const latlngs = road.points.map(p => L.latLng(p.lat, p.lng));
      const poly = L.polyline(latlngs, {
        color: '#3b82f6',
        weight: Math.max(3, road.width / 3),
        opacity: 0.8,
      }).addTo(map);
      poly.bindPopup(`<b>${road.name}</b><br/>ID: ${road.id}<br/>Width: ${road.width}m`);
      layers.roads[road.id] = poly;
    });

    // Render spawns (green markers)
    data.spawns.forEach(spawn => {
      const icon = L.divIcon({
        className: 'custom-marker',
        html: `<div style="background:#22c55e;width:20px;height:20px;border-radius:50%;border:2px solid white;display:flex;align-items:center;justify-content:center;color:white;font-size:10px;font-weight:bold;">S</div>`,
        iconSize: [20, 20],
        iconAnchor: [10, 10],
      });
      const marker = L.marker([spawn.lat, spawn.lng], { icon }).addTo(map);
      marker.bindPopup(`<b>${spawn.name}</b><br/>ID: ${spawn.id}<br/>Rate: ${spawn.rate}/s`);
      layers.spawns[spawn.id] = marker;
    });

    // Render choke points (yellow circles)
    data.chokePoints.forEach(cp => {
      const circle = L.circle([cp.lat, cp.lng], {
        radius: cp.radius,
        color: '#eab308',
        fillColor: '#eab308',
        fillOpacity: 0.2,
        weight: 2,
      }).addTo(map);
      circle.bindPopup(`<b>${cp.name}</b><br/>ID: ${cp.id}<br/>Radius: ${cp.radius}m`);
      layers.chokePoints[cp.id] = circle;
    });

    // Render exits (red markers)
    data.exits.forEach(exit => {
      const marker = L.circleMarker([exit.lat, exit.lng], {
        radius: 8,
        color: '#ef4444',
        fillColor: '#ef4444',
        fillOpacity: 0.8,
        weight: 2,
      }).addTo(map);
      marker.bindPopup(`<b>${exit.name}</b><br/>ID: ${exit.id}`);
      layers.exits[exit.id] = marker;
    });

    // Render origin (purple marker)
    const originIcon = L.divIcon({
      className: 'custom-marker',
      html: `<div style="background:#8b5cf6;width:24px;height:24px;border-radius:50%;border:3px solid white;display:flex;align-items:center;justify-content:center;color:white;font-size:12px;font-weight:bold;">O</div>`,
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });
    layers.origin = L.marker([data.origin.lat, data.origin.lng], { icon: originIcon }).addTo(map);
    layers.origin.bindPopup(`<b>Origin</b><br/>lat: ${data.origin.lat.toFixed(6)}<br/>lng: ${data.origin.lng.toFixed(6)}`);

  }, [data]);

  // Render road preview
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const layers = layersRef.current;
    if (layers.preview) {
      map.removeLayer(layers.preview);
      layers.preview = null;
    }

    if (currentRoad && currentRoad.points.length > 0) {
      const latlngs = currentRoad.points.map(p => L.latLng(p.lat, p.lng));
      layers.preview = L.polyline(latlngs, {
        color: '#f97316',
        weight: 4,
        dashArray: '10, 10',
        opacity: 0.8,
      }).addTo(map);
    }
  }, [currentRoad]);

  // Road drawing controls
  const startRoad = () => {
    setCurrentRoad({
      id: uid('road_'),
      name: `Road ${data.roads.length + 1}`,
      width: 15,
      points: [],
    });
    setMode('road');
  };

  const finishRoad = () => {
    if (!currentRoad || currentRoad.points.length < 2) {
      alert('A road needs at least 2 points');
      return;
    }
    setData(prev => ({ ...prev, roads: [...prev.roads, currentRoad] }));
    setCurrentRoad(null);
    setMode('pan');
  };

  const cancelRoad = () => {
    setCurrentRoad(null);
    setMode('pan');
  };

  // YAML panel
  const openYamlPanel = () => {
    setYamlText(exportToYAML(data));
    setShowYamlPanel(true);
  };

  const applyYaml = () => {
    const imported = importFromYAML(yamlText);
    if (imported) {
      setData(imported);
      setShowYamlPanel(false);
      // Recenter map
      if (mapRef.current) {
        mapRef.current.setView([imported.origin.lat, imported.origin.lng], 16);
      }
    } else {
      alert('Failed to parse YAML');
    }
  };

  const downloadYaml = () => {
    const text = exportToYAML(data);
    const blob = new Blob([text], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${data.venue.name.toLowerCase().replace(/\s+/g, '_')}.yaml`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const loadYamlFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const text = reader.result as string;
      const imported = importFromYAML(text);
      if (imported) {
        setData(imported);
        if (mapRef.current) {
          mapRef.current.setView([imported.origin.lat, imported.origin.lng], 16);
        }
      } else {
        alert('Failed to parse YAML file');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  // Delete item
  const deleteItem = (type: string, id: string) => {
    if (type === 'road') {
      setData(prev => ({ ...prev, roads: prev.roads.filter(r => r.id !== id) }));
    } else if (type === 'spawn') {
      setData(prev => ({ ...prev, spawns: prev.spawns.filter(s => s.id !== id) }));
    } else if (type === 'choke') {
      setData(prev => ({ ...prev, chokePoints: prev.chokePoints.filter(c => c.id !== id) }));
    } else if (type === 'exit') {
      setData(prev => ({ ...prev, exits: prev.exits.filter(e => e.id !== id) }));
    }
  };

  const modeButtons: { mode: EditorMode; label: string; color: string }[] = [
    { mode: 'pan', label: '🖐️ Pan', color: 'gray' },
    { mode: 'road', label: '🛣️ Road', color: 'blue' },
    { mode: 'spawn', label: '🟢 Spawn', color: 'green' },
    { mode: 'choke', label: '🟡 Choke', color: 'yellow' },
    { mode: 'exit', label: '🔴 Exit', color: 'red' },
    { mode: 'origin', label: '🟣 Origin', color: 'purple' },
    { mode: 'erase', label: '🗑️ Erase', color: 'gray' },
  ];

  return (
    <div className="h-screen w-screen flex flex-col bg-gray-900 text-white">
      {/* Header */}
      <header className="flex items-center gap-4 px-4 py-2 bg-gray-800 border-b border-gray-700">
        <h1 className="text-lg font-bold">Venue Editor</h1>
        
        {/* Mode buttons */}
        <div className="flex gap-1">
          {modeButtons.map(btn => (
            <button
              key={btn.mode}
              onClick={() => setMode(btn.mode)}
              disabled={btn.mode === 'road' && currentRoad !== null}
              className={`px-3 py-1 text-sm rounded transition ${
                mode === btn.mode
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 hover:bg-gray-600'
              } disabled:opacity-50`}
            >
              {btn.label}
            </button>
          ))}
        </div>

        {/* Road controls */}
        {currentRoad ? (
          <div className="flex gap-2 ml-4">
            <span className="text-sm text-orange-400">
              Drawing: {currentRoad.points.length} points
            </span>
            <button
              onClick={finishRoad}
              className="px-3 py-1 text-sm bg-green-600 hover:bg-green-500 rounded"
            >
              ✓ Finish Road
            </button>
            <button
              onClick={cancelRoad}
              className="px-3 py-1 text-sm bg-red-600 hover:bg-red-500 rounded"
            >
              ✕ Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={startRoad}
            className="px-3 py-1 text-sm bg-blue-600 hover:bg-blue-500 rounded ml-4"
          >
            + New Road
          </button>
        )}

        <div className="flex-1" />

        {/* File controls */}
        <label className="px-3 py-1 text-sm bg-gray-700 hover:bg-gray-600 rounded cursor-pointer">
          📂 Load YAML
          <input
            type="file"
            accept=".yaml,.yml"
            onChange={loadYamlFile}
            className="hidden"
          />
        </label>
        <button
          onClick={downloadYaml}
          className="px-3 py-1 text-sm bg-gray-700 hover:bg-gray-600 rounded"
        >
          💾 Save YAML
        </button>
        <button
          onClick={openYamlPanel}
          className="px-3 py-1 text-sm bg-gray-700 hover:bg-gray-600 rounded"
        >
          📝 View YAML
        </button>
        <Link
          to="/"
          className="px-3 py-1 text-sm bg-purple-600 hover:bg-purple-500 rounded"
        >
          ← Back to Simulation
        </Link>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-72 bg-gray-800 border-r border-gray-700 overflow-y-auto p-3">
          {/* Venue Info */}
          <div className="mb-4">
            <h2 className="text-sm font-semibold text-gray-400 uppercase mb-2">Venue Info</h2>
            <input
              type="text"
              value={data.venue.name}
              onChange={e => setData(prev => ({ ...prev, venue: { ...prev.venue, name: e.target.value } }))}
              placeholder="Venue Name"
              className="w-full px-2 py-1 bg-gray-700 rounded text-sm mb-2"
            />
            <input
              type="text"
              value={data.venue.location}
              onChange={e => setData(prev => ({ ...prev, venue: { ...prev.venue, location: e.target.value } }))}
              placeholder="Location"
              className="w-full px-2 py-1 bg-gray-700 rounded text-sm mb-2"
            />
            <textarea
              value={data.venue.description}
              onChange={e => setData(prev => ({ ...prev, venue: { ...prev.venue, description: e.target.value } }))}
              placeholder="Description"
              className="w-full px-2 py-1 bg-gray-700 rounded text-sm resize-none h-16"
            />
          </div>

          {/* Origin */}
          <div className="mb-4">
            <h2 className="text-sm font-semibold text-gray-400 uppercase mb-2">
              Origin <span className="text-purple-400">🟣</span>
            </h2>
            <div className="text-xs text-gray-300">
              lat: {data.origin.lat.toFixed(6)}, lng: {data.origin.lng.toFixed(6)}
            </div>
          </div>

          {/* Roads */}
          <div className="mb-4">
            <h2 className="text-sm font-semibold text-gray-400 uppercase mb-2">
              Roads ({data.roads.length}) <span className="text-blue-400">🛣️</span>
            </h2>
            {data.roads.map(road => (
              <div key={road.id} className="flex items-center justify-between bg-gray-700 rounded px-2 py-1 mb-1 text-sm">
                <span className="truncate flex-1">{road.name}</span>
                <span className="text-gray-400 text-xs mx-2">{road.points.length}pts</span>
                <button
                  onClick={() => deleteItem('road', road.id)}
                  className="text-red-400 hover:text-red-300"
                >
                  ×
                </button>
              </div>
            ))}
          </div>

          {/* Spawn Points */}
          <div className="mb-4">
            <h2 className="text-sm font-semibold text-gray-400 uppercase mb-2">
              Spawns ({data.spawns.length}) <span className="text-green-400">🟢</span>
            </h2>
            {data.spawns.map(spawn => (
              <div key={spawn.id} className="flex items-center justify-between bg-gray-700 rounded px-2 py-1 mb-1 text-sm">
                <span className="truncate flex-1">{spawn.name}</span>
                <button
                  onClick={() => deleteItem('spawn', spawn.id)}
                  className="text-red-400 hover:text-red-300"
                >
                  ×
                </button>
              </div>
            ))}
          </div>

          {/* Choke Points */}
          <div className="mb-4">
            <h2 className="text-sm font-semibold text-gray-400 uppercase mb-2">
              Choke Points ({data.chokePoints.length}) <span className="text-yellow-400">🟡</span>
            </h2>
            {data.chokePoints.map(cp => (
              <div key={cp.id} className="flex items-center justify-between bg-gray-700 rounded px-2 py-1 mb-1 text-sm">
                <span className="truncate flex-1">{cp.name}</span>
                <span className="text-gray-400 text-xs mx-2">{cp.radius}m</span>
                <button
                  onClick={() => deleteItem('choke', cp.id)}
                  className="text-red-400 hover:text-red-300"
                >
                  ×
                </button>
              </div>
            ))}
          </div>

          {/* Exits */}
          <div className="mb-4">
            <h2 className="text-sm font-semibold text-gray-400 uppercase mb-2">
              Exits ({data.exits.length}) <span className="text-red-400">🔴</span>
            </h2>
            {data.exits.map(exit => (
              <div key={exit.id} className="flex items-center justify-between bg-gray-700 rounded px-2 py-1 mb-1 text-sm">
                <span className="truncate flex-1">{exit.name}</span>
                <button
                  onClick={() => deleteItem('exit', exit.id)}
                  className="text-red-400 hover:text-red-300"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </aside>

        {/* Map */}
        <div className="flex-1 relative">
          <div ref={mapContainerRef} className="absolute inset-0" />
          
          {/* Mode indicator */}
          <div className="absolute top-4 left-4 bg-gray-900/90 px-3 py-2 rounded-lg z-[1000]">
            <span className="text-sm text-gray-400">Mode: </span>
            <span className="text-sm font-medium capitalize">{mode}</span>
          </div>

          {/* Instructions */}
          <div className="absolute bottom-4 left-4 bg-gray-900/90 px-3 py-2 rounded-lg z-[1000] text-sm text-gray-300">
            {mode === 'pan' && 'Click and drag to pan the map'}
            {mode === 'road' && (currentRoad ? 'Click to add road points, then "Finish Road"' : 'Click "New Road" to start drawing')}
            {mode === 'spawn' && 'Click to place spawn points (where agents enter)'}
            {mode === 'choke' && 'Click to place choke points (monitoring zones)'}
            {mode === 'exit' && 'Click to place exits (where agents leave)'}
            {mode === 'origin' && 'Click to set the coordinate origin point'}
            {mode === 'erase' && 'Click near a feature to delete it'}
          </div>
        </div>
      </div>

      {/* YAML Panel */}
      {showYamlPanel && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[2000]">
          <div className="bg-gray-800 rounded-lg w-[800px] max-h-[90vh] flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
              <h2 className="text-lg font-semibold">YAML Configuration</h2>
              <button
                onClick={() => setShowYamlPanel(false)}
                className="text-gray-400 hover:text-white text-xl"
              >
                ×
              </button>
            </div>
            <textarea
              value={yamlText}
              onChange={e => setYamlText(e.target.value)}
              className="flex-1 m-4 p-3 bg-gray-900 rounded font-mono text-sm resize-none"
              spellCheck={false}
            />
            <div className="flex gap-3 px-4 py-3 border-t border-gray-700 justify-end">
              <button
                onClick={applyYaml}
                className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded"
              >
                Apply Changes
              </button>
              <button
                onClick={() => {
                  const blob = new Blob([yamlText], { type: 'text/yaml' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = 'venue.yaml';
                  a.click();
                  URL.revokeObjectURL(url);
                }}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded"
              >
                Download
              </button>
              <button
                onClick={() => setShowYamlPanel(false)}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
