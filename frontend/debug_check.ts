// Quick test to check coordinate conversion
const METERS_PER_DEG_LAT = 111320;

const origin = { lat: 13.05, lng: 80.2824 };
const latRad = (origin.lat * Math.PI) / 180;
const metersPerDegLng = METERS_PER_DEG_LAT * Math.cos(latRad);

console.log('Origin:', origin);
console.log('metersPerDegLng:', metersPerDegLng);

// Test agent at spawn_north: x=65.07m, y=222.64m
const agentSim = { x: 65.07, y: 222.64 };

// Convert to geo
const agentGeo = {
  lat: origin.lat + agentSim.y / METERS_PER_DEG_LAT,
  lng: origin.lng + agentSim.x / metersPerDegLng
};

console.log('Agent sim coords:', agentSim);
console.log('Agent geo coords:', agentGeo);
console.log('Expected: lat=13.052000, lng=80.283000');
