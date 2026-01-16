"""
FastAPI application for the crowd simulation backend.
Provides WebSocket for real-time state and REST endpoints for control.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import List, Set
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.types import (
    GeoPoint, ChokePoint, SimulationConfig, ControlMessage,
    SimulationState, CircleGeometry, SensorConfig, SensorType
)
from simulation.engine import SimulationEngine


# ============================================================================
# Application State
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.add(connection)
        
        # Clean up disconnected
        for conn in disconnected:
            self.active_connections.discard(conn)


# Global instances
manager = ConnectionManager()
engine: SimulationEngine = None
simulation_task: asyncio.Task = None


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - start/stop simulation."""
    global engine, simulation_task
    
    # Initialize engine
    engine = SimulationEngine()
    
    # Set up state broadcast callback
    async def broadcast_state(state: SimulationState):
        if manager.active_connections:
            await manager.broadcast(state.model_dump_json())
    
    engine.on_state_update = broadcast_state
    
    # Start simulation loop
    simulation_task = asyncio.create_task(engine.run_loop())
    
    yield
    
    # Cleanup
    if simulation_task:
        simulation_task.cancel()
        try:
            await simulation_task
        except asyncio.CancelledError:
            pass


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Crowd Simulation API",
    description="Real-time crowd instability simulation for early warning systems",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API root - health check."""
    return {"status": "ok", "service": "crowd-simulation"}


@app.get("/config")
async def get_config() -> SimulationConfig:
    """Get current simulation configuration."""
    return engine.config


@app.post("/config")
async def update_config(updates: dict):
    """Update simulation configuration."""
    engine.update_config(**updates)
    return {"status": "updated", "config": engine.config}


@app.post("/control/start")
async def start_simulation():
    """Start the simulation."""
    engine.config.running = True
    return {"status": "running"}


@app.post("/control/stop")
async def stop_simulation():
    """Stop/pause the simulation."""
    engine.config.running = False
    return {"status": "stopped"}


@app.post("/control/reset")
async def reset_simulation():
    """Reset simulation to initial state."""
    engine.reset()
    return {"status": "reset"}


class SetOriginRequest(BaseModel):
    lat: float
    lng: float


@app.post("/map/origin")
async def set_map_origin(request: SetOriginRequest):
    """Set the map origin point."""
    engine.set_origin(GeoPoint(lat=request.lat, lng=request.lng))
    return {"status": "origin_set", "origin": {"lat": request.lat, "lng": request.lng}}


@app.get("/choke-points")
async def get_choke_points() -> List[ChokePoint]:
    """Get all choke points."""
    return engine.choke_points


class AddChokePointRequest(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    radius: float = 15.0


@app.post("/choke-points")
async def add_choke_point(request: AddChokePointRequest):
    """Add a new choke point."""
    cp = ChokePoint(
        id=request.id,
        name=request.name,
        center=GeoPoint(lat=request.lat, lng=request.lng),
        geometry=CircleGeometry(radius=request.radius),
        sensors=[
            SensorConfig(type=SensorType.MMWAVE, enabled=True),
            SensorConfig(type=SensorType.AUDIO, enabled=True),
            SensorConfig(type=SensorType.CAMERA, enabled=True),
        ]
    )
    engine.add_choke_point(cp)
    return {"status": "added", "choke_point": cp}


@app.delete("/choke-points/{choke_point_id}")
async def remove_choke_point(choke_point_id: str):
    """Remove a choke point."""
    engine.remove_choke_point(choke_point_id)
    return {"status": "removed", "id": choke_point_id}


# ============================================================================
# Venue Management
# ============================================================================

@app.get("/venues")
async def list_venues():
    """Get list of available venue configurations."""
    venues = engine.get_available_venues()
    return {"venues": venues}


class LoadVenueRequest(BaseModel):
    venue_id: str


@app.post("/venues/load")
async def load_venue(request: LoadVenueRequest):
    """Load a venue configuration by ID."""
    success = engine.load_venue(request.venue_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Venue '{request.venue_id}' not found")
    
    venue_info = engine.get_current_venue_info()
    return {
        "status": "loaded",
        "venue": {
            "id": venue_info.id,
            "name": venue_info.name,
            "location": venue_info.location,
            "description": venue_info.description,
            "spawn_point_count": venue_info.spawn_point_count,
            "choke_point_count": venue_info.choke_point_count
        }
    }


@app.get("/venues/current")
async def get_current_venue():
    """Get information about the currently loaded venue."""
    venue_info = engine.get_current_venue_info()
    if venue_info is None:
        return {
            "venue": None,
            "message": "No venue loaded (using demo mode)"
        }
    
    return {
        "venue": {
            "id": venue_info.id,
            "name": venue_info.name,
            "location": venue_info.location,
            "description": venue_info.description,
            "roads": venue_info.roads,
            "spawn_point_count": venue_info.spawn_point_count,
            "choke_point_count": venue_info.choke_point_count,
            "origin": venue_info.origin
        }
    }


# ============================================================================
# Perturbation Controls
# ============================================================================

@app.post("/perturbation/opposing-flow")
async def toggle_opposing_flow(enabled: bool = True):
    """Enable/disable opposing crowd flow."""
    engine.update_config(opposing_flow_enabled=enabled)
    return {"status": "updated", "opposing_flow_enabled": enabled}


@app.post("/perturbation/density-surge")
async def toggle_density_surge(enabled: bool = True):
    """Enable/disable density surge (doubles inflow)."""
    engine.update_config(density_surge_enabled=enabled)
    return {"status": "updated", "density_surge_enabled": enabled}


@app.post("/perturbation/inflow")
async def set_inflow_multiplier(multiplier: float = 1.0):
    """Set crowd inflow multiplier (0.0 - 5.0)."""
    multiplier = max(0.0, min(5.0, multiplier))
    engine.update_config(current_inflow_multiplier=multiplier)
    return {"status": "updated", "inflow_multiplier": multiplier}


@app.post("/perturbation/speed")
async def set_speed_multiplier(multiplier: float = 1.0):
    """Set simulation speed multiplier (0.1 - 10.0)."""
    multiplier = max(0.1, min(10.0, multiplier))
    engine.update_config(speed_multiplier=multiplier)
    return {"status": "updated", "speed_multiplier": multiplier}


# ============================================================================
# WebSocket
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time state updates.
    
    Receives: ControlMessage objects
    Sends: SimulationState objects at ~30 Hz
    """
    await manager.connect(websocket)
    
    try:
        # Send initial state
        state = engine.get_state()
        await websocket.send_text(state.model_dump_json())
        
        # Handle incoming control messages
        while True:
            try:
                data = await websocket.receive_text()
                msg = ControlMessage.model_validate_json(data)
                
                # Process control message
                await handle_control_message(msg)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                # Send error back to client
                await websocket.send_json({"error": str(e)})
    
    finally:
        manager.disconnect(websocket)


async def handle_control_message(msg: ControlMessage):
    """Handle control messages from WebSocket clients."""
    
    if msg.action == "start":
        engine.config.running = True
    
    elif msg.action == "stop":
        engine.config.running = False
    
    elif msg.action == "reset":
        engine.reset()
    
    elif msg.action == "set_speed":
        if msg.payload and "multiplier" in msg.payload:
            engine.update_config(speed_multiplier=msg.payload["multiplier"])
    
    elif msg.action == "set_inflow":
        if msg.payload and "multiplier" in msg.payload:
            engine.update_config(current_inflow_multiplier=msg.payload["multiplier"])
    
    elif msg.action == "toggle_opposing_flow":
        enabled = msg.payload.get("enabled", True) if msg.payload else True
        engine.update_config(opposing_flow_enabled=enabled)
    
    elif msg.action == "toggle_density_surge":
        enabled = msg.payload.get("enabled", True) if msg.payload else True
        engine.update_config(density_surge_enabled=enabled)
    
    elif msg.action == "add_choke_point":
        if msg.payload:
            cp = ChokePoint(
                id=msg.payload["id"],
                name=msg.payload.get("name", msg.payload["id"]),
                center=GeoPoint(
                    lat=msg.payload["lat"],
                    lng=msg.payload["lng"]
                ),
                geometry=CircleGeometry(radius=msg.payload.get("radius", 15)),
                sensors=[
                    SensorConfig(type=SensorType.MMWAVE, enabled=True),
                    SensorConfig(type=SensorType.AUDIO, enabled=True),
                    SensorConfig(type=SensorType.CAMERA, enabled=True),
                ]
            )
            engine.add_choke_point(cp)
    
    elif msg.action == "remove_choke_point":
        if msg.payload and "id" in msg.payload:
            engine.remove_choke_point(msg.payload["id"])
    
    elif msg.action == "load_venue":
        if msg.payload and "venue_id" in msg.payload:
            engine.load_venue(msg.payload["venue_id"])


# ============================================================================
# Run with uvicorn
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
