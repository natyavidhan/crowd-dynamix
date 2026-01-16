"""
Tests for REST API endpoints.
Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app


@pytest.fixture
def client():
    """Create a test client."""
    with TestClient(app) as c:
        yield c


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_root_returns_ok(self, client):
        """GET / should return status ok."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "crowd-simulation"


class TestConfigEndpoints:
    """Test configuration endpoints."""
    
    def test_get_config(self, client):
        """GET /config should return simulation config."""
        response = client.get("/config")
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        assert "running" in data
        assert "speed_multiplier" in data
        assert "base_inflow_rate" in data
        assert "current_inflow_multiplier" in data
        assert "max_agents" in data
        assert "opposing_flow_enabled" in data
        assert "density_surge_enabled" in data
    
    def test_update_config(self, client):
        """POST /config should update config."""
        response = client.post("/config", json={"speed_multiplier": 2.0})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"
        assert data["config"]["speed_multiplier"] == 2.0
    
    def test_update_config_invalid_field_ignored(self, client):
        """POST /config with invalid field should not crash."""
        response = client.post("/config", json={"invalid_field": 123})
        assert response.status_code == 200


class TestControlEndpoints:
    """Test simulation control endpoints."""
    
    def test_start_simulation(self, client):
        """POST /control/start should start simulation."""
        response = client.post("/control/start")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        
        # Verify config updated
        config = client.get("/config").json()
        assert config["running"] == True
    
    def test_stop_simulation(self, client):
        """POST /control/stop should stop simulation."""
        # Start first
        client.post("/control/start")
        
        response = client.post("/control/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
        
        # Verify config updated
        config = client.get("/config").json()
        assert config["running"] == False
    
    def test_reset_simulation(self, client):
        """POST /control/reset should reset simulation."""
        response = client.post("/control/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reset"


class TestMapEndpoints:
    """Test map configuration endpoints."""
    
    def test_set_map_origin(self, client):
        """POST /map/origin should set origin."""
        response = client.post("/map/origin", json={
            "lat": 13.0827,
            "lng": 80.2707
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "origin_set"
        assert data["origin"]["lat"] == 13.0827
        assert data["origin"]["lng"] == 80.2707


class TestChokePointEndpoints:
    """Test choke point CRUD endpoints."""
    
    def test_get_choke_points(self, client):
        """GET /choke-points should return list."""
        response = client.get("/choke-points")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_add_choke_point(self, client):
        """POST /choke-points should add new choke point."""
        response = client.post("/choke-points", json={
            "id": "test_cp_1",
            "name": "Test Choke Point",
            "lat": 12.9716,
            "lng": 77.5946,
            "radius": 20.0
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "added"
        assert data["choke_point"]["id"] == "test_cp_1"
        assert data["choke_point"]["name"] == "Test Choke Point"
    
    def test_remove_choke_point(self, client):
        """DELETE /choke-points/{id} should remove choke point."""
        # First add one
        client.post("/choke-points", json={
            "id": "test_cp_delete",
            "name": "To Delete",
            "lat": 12.9716,
            "lng": 77.5946,
            "radius": 15.0
        })
        
        # Then delete
        response = client.delete("/choke-points/test_cp_delete")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "removed"
        assert data["id"] == "test_cp_delete"


class TestPerturbationEndpoints:
    """Test perturbation control endpoints."""
    
    def test_toggle_opposing_flow_on(self, client):
        """POST /perturbation/opposing-flow should enable opposing flow."""
        response = client.post("/perturbation/opposing-flow?enabled=true")
        assert response.status_code == 200
        data = response.json()
        assert data["opposing_flow_enabled"] == True
    
    def test_toggle_opposing_flow_off(self, client):
        """POST /perturbation/opposing-flow should disable opposing flow."""
        response = client.post("/perturbation/opposing-flow?enabled=false")
        assert response.status_code == 200
        data = response.json()
        assert data["opposing_flow_enabled"] == False
    
    def test_toggle_density_surge_on(self, client):
        """POST /perturbation/density-surge should enable density surge."""
        response = client.post("/perturbation/density-surge?enabled=true")
        assert response.status_code == 200
        data = response.json()
        assert data["density_surge_enabled"] == True
    
    def test_toggle_density_surge_off(self, client):
        """POST /perturbation/density-surge should disable density surge."""
        response = client.post("/perturbation/density-surge?enabled=false")
        assert response.status_code == 200
        data = response.json()
        assert data["density_surge_enabled"] == False
    
    def test_set_inflow_multiplier(self, client):
        """POST /perturbation/inflow should set multiplier."""
        response = client.post("/perturbation/inflow?multiplier=2.5")
        assert response.status_code == 200
        data = response.json()
        assert data["inflow_multiplier"] == 2.5
    
    def test_set_inflow_multiplier_clamped(self, client):
        """POST /perturbation/inflow should clamp to valid range."""
        # Test upper bound
        response = client.post("/perturbation/inflow?multiplier=10.0")
        assert response.status_code == 200
        data = response.json()
        assert data["inflow_multiplier"] == 5.0  # Max is 5.0
        
        # Test lower bound
        response = client.post("/perturbation/inflow?multiplier=-1.0")
        assert response.status_code == 200
        data = response.json()
        assert data["inflow_multiplier"] == 0.0  # Min is 0.0
    
    def test_set_speed_multiplier(self, client):
        """POST /perturbation/speed should set multiplier."""
        response = client.post("/perturbation/speed?multiplier=3.0")
        assert response.status_code == 200
        data = response.json()
        assert data["speed_multiplier"] == 3.0
    
    def test_set_speed_multiplier_clamped(self, client):
        """POST /perturbation/speed should clamp to valid range."""
        # Test upper bound
        response = client.post("/perturbation/speed?multiplier=20.0")
        assert response.status_code == 200
        data = response.json()
        assert data["speed_multiplier"] == 10.0  # Max is 10.0
        
        # Test lower bound
        response = client.post("/perturbation/speed?multiplier=0.01")
        assert response.status_code == 200
        data = response.json()
        assert data["speed_multiplier"] == 0.1  # Min is 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
